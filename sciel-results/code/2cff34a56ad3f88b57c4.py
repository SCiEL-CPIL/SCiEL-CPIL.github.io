import numpy as np
from autoresearch_suite.chem_clusters import relax
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading

# Global variables for thread-safe exchange
lock = threading.Lock()
# We will manage the parallel BH manually in threads to allow synchronization for exchange

class BHChain:
    def __init__(self, initial_pos, potential, seed, sigma=0.2, temperature=1.0):
        self.rng = np.random.default_rng(seed)
        self.pos = initial_pos.copy()
        try:
            self.energy, self.pos = relax(self.pos, potential)
        except Exception:
            self.energy = np.inf
        self.best_energy = self.energy
        self.best_pos = self.pos.copy()
        self.potential = potential
        self.sigma = sigma
        self.temperature = temperature
        self.accepted = 0
        self.total = 0

    def step(self):
        self.total += 1
        noise = self.rng.normal(0, self.sigma, self.pos.shape)
        new_pos = self.pos + noise
        try:
            new_energy, new_pos = relax(new_pos, self.potential)
        except Exception:
            return

        delta_e = new_energy - self.energy
        if delta_e < 0 or self.rng.random() < np.exp(-delta_e / self.temperature):
            self.pos = new_pos
            self.energy = new_energy
            self.accepted += 1
            if self.energy < self.best_energy:
                self.best_energy = self.energy
                self.best_pos = self.pos.copy()

    def adapt_sigma(self):
        if self.total > 10:
            rate = self.accepted / self.total
            if rate < 0.2:
                self.sigma *= 1.2
            elif rate > 0.8:
                self.sigma *= 0.8
            self.accepted = 0
            self.total = 0

def generate_diverse_initials(n, num_initials=4):
    """Generate diverse initial configurations."""
    initials = []
    # 1. FCC
    a = 1.05 * np.sqrt(2)
    points = []
    max_coord = int(n ** (1/3) * a * 2) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) < (n ** (1/3)) * a * 0.8:
                        points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    initials.append(points[indices])

    # 2. Icosahedron (if n <= 55)
    if n <= 55:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        scale = 1.05 / 2.0
        verts = verts * scale
        center = np.array([0, 0, 0])
        edge_midpoints = []
        for i in range(12):
            for j in range(i+1, 12):
                if np.isclose(np.linalg.norm(verts[i] - verts[j]), 1.05):
                    edge_midpoints.append((verts[i] + verts[j]) / 2.0)
        edge_midpoints = np.array(edge_midpoints)
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                if np.isclose(np.linalg.norm(verts[i] - verts[j]), 1.05):
                    for k in range(j+1, 12):
                        if (np.isclose(np.linalg.norm(verts[i] - verts[k]), 1.05) and
                            np.isclose(np.linalg.norm(verts[j] - verts[k]), 1.05)):
                            face_centers.append((verts[i] + verts[j] + verts[k]) / 3.0)
        face_centers = np.array(face_centers)
        all_atoms = np.vstack([center[None, :], verts, edge_midpoints, face_centers])
        dists = np.linalg.norm(all_atoms, axis=1)
        indices = np.argsort(dists)[:n]
        initials.append(all_atoms[indices])
    else:
        # For n > 55, use a distorted FCC or random
        rng = np.random.default_rng(100)
        pos = initials[0] + rng.normal(0, 0.1, (n, 3))
        initials.append(pos)

    # 3 & 4. Random Close Packing
    for seed in [42, 123]:
        rng = np.random.default_rng(seed)
        L = 1.05
        radius = L * n ** (1/3) * 1.2
        points = []
        attempts = 0
        max_attempts = 50000
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if points:
                    dists = np.linalg.norm(np.array(points) - p, axis=1)
                    if np.min(dists) >= 0.9 * L:
                        points.append(p)
                        attempts = 0
                else:
                    points.append(p)
                    attempts = 0
                if len(points) % 100 == 0:
                    attempts += 1
        if len(points) >= n:
            initials.append(np.array(points[:n]))
        else:
            # Fallback
            initials.append(rng.uniform(-5, 5, (n, 3)))

    return initials

def construct(instance):
    n = instance['n']
    potential = instance['potential']
    
    # Generate diverse initial candidates
    initials = generate_diverse_initials(n, 4)
    
    # Relax them initially
    relaxed = []
    for pos in initials:
        try:
            e, p = relax(pos, potential)
            relaxed.append((e, p))
        except Exception:
            pass
            
    if not relaxed:
        return np.random.default_rng(0).uniform(-5, 5, (n, 3))

    # Sort by energy
    relaxed.sort(key=lambda x: x[0])
    
    # Initialize BH Chains
    num_chains = 4
    chains = []
    for i in range(min(num_chains, len(relaxed))):
        e, p = relaxed[i]
        chains.append(BHChain(p, potential, seed=i, sigma=0.2, temperature=1.0))
        
    # If we have fewer than 4, duplicate best
    while len(chains) < num_chains:
        best = min(chains, key=lambda c: c.energy)
        new_chain = BHChain(best.pos, potential, seed=len(chains), sigma=0.2, temperature=1.0)
        chains.append(new_chain)

    # Parallel BH with Exchange
    steps_per_round = 20
    num_rounds = 5
    exchange_interval = 5 # every 5 rounds? Or every round?
    
    # We'll use a simple loop with threads for synchronization
    # Since we need to exchange, we must synchronize.
    
    best_global_energy = min(c.best_energy for c in chains)
    best_global_pos = min(chains, key=lambda c: c.best_energy).best_pos
    
    for round_num in range(num_rounds):
        # Run steps in parallel
        futures = []
        with ThreadPoolExecutor(max_workers=num_chains) as executor:
            for i, chain in enumerate(chains):
                futures.append(executor.submit(chain.step))
            for f in futures:
                f.result()
                
        # Adapt sigma
        for chain in chains:
            chain.adapt_sigma()
            
        # Exchange
        if (round_num + 1) % exchange_interval == 0:
            # Random pair exchange
            i, j = np.random.choice(num_chains, 2, replace=False)
            c1, c2 = chains[i], chains[j]
            e1, e2 = c1.energy, c2.energy
            # Metropolis criterion for exchange
            # If e1 > e2, swap with prob 1. Else exp(-(e1-e2)/T_exch)
            # T_exch can be related to the temperatures of the chains or a fixed value
            T_exch = 1.0 
            delta = e1 - e2
            if delta < 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T_exch)
                
            if np.random.random() < prob:
                # Swap current states
                c1.pos, c2.pos = c2.pos.copy(), c1.pos.copy()
                c1.energy, c2.energy = c2.energy, c1.energy
                # Note: best_pos/best_energy are not swapped, they stay with the chain
                # This is correct for parallel tempering: the "best" found by a chain is kept.
                # But usually, we swap the *current* state.
                
        # Update global best
        for chain in chains:
            if chain.best_energy < best_global_energy:
                best_global_energy = chain.best_energy
                best_global_pos = chain.best_pos.copy()

    # Final relaxation of the best candidate
    try:
        e_final, pos_final = relax(best_global_pos, potential)
        if e_final < best_global_energy:
            best_global_pos = pos_final
    except Exception:
        pass

    return best_global_pos
