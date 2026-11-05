import time
import numpy as np
import random
import math

def construct():
    n = 256
    deadline = time.time() + 110  # safety margin for 120s limit
    
    # Precompute all 4-term APs as numpy arrays for vectorized operations
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    ap_array = np.array(ap_list, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, precompute which APs it belongs to and its index within each AP
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for j, pos in enumerate(ap):
            pos_aps[pos].append((idx, j))
    
    # Thue-Morse initialization
    def thue_morse(i):
        return bin(i).count('1') % 2
    
    init_colors = np.array([thue_morse(i) for i in range(n)], dtype=np.int8)
    
    # Count mono APs using vectorized operations
    def count_mono_vec(colors):
        # colors shape (n,), ap_array shape (num_aps, 4)
        # Get colors for each term in each AP
        ap_colors = colors[ap_array]  # shape (num_aps, 4)
        # Check if all 4 are equal
        mono = np.all(ap_colors == ap_colors[:, 0:1], axis=1)
        return int(np.sum(mono))
    
    # Number of replicas for parallel tempering
    num_replicas = 8
    # Temperatures for each replica (geometric progression)
    T_base = 5.0
    T_min = 0.05
    temps = np.geomspace(T_base, T_min, num_replicas)
    
    # Initialize replicas with slightly different starts
    rng = np.random.RandomState(42)
    replicas = np.zeros((num_replicas, n), dtype=np.int8)
    replica_energies = np.zeros(num_replicas, dtype=np.float64)
    
    for r in range(num_replicas):
        replicas[r] = init_colors.copy()
        if r > 0:
            # Perturb each replica slightly
            num_flips = r * 3
            flip_positions = rng.choice(n, size=num_flips, replace=False)
            replicas[r, flip_positions] = 1 - replicas[r, flip_positions]
        replica_energies[r] = count_mono_vec(replicas[r])
    
    # Track best
    best_energy = int(replica_energies.min())
    best_idx = int(np.argmin(replica_energies))
    best_colors = replicas[best_idx].copy()
    
    # Precompute for fast delta calculation
    # For each position, store the AP indices and positions
    pos_ap_indices = []
    pos_ap_positions = []
    for pos in range(n):
        idxs = [x[0] for x in pos_aps[pos]]
        posns = [x[1] for x in pos_aps[pos]]
        pos_ap_indices.append(np.array(idxs, dtype=np.int32))
        pos_ap_positions.append(np.array(posns, dtype=np.int32))
    
    # Parallel tempering main loop
    swap_interval = 5  # Try swap every this many iterations
    iter_count = 0
    
    while time.time() < deadline:
        # Update all replicas in parallel (vectorized where possible)
        for r in range(num_replicas):
            T = temps[r]
            # Pick a random position to flip
            pos = rng.randint(0, n - 1)
            old_color = replicas[r, pos]
            new_color = 1 - old_color
            
            # Compute delta
            delta = 0
            ap_idxs = pos_ap_indices[pos]
            ap_posns = pos_ap_positions[pos]
            
            for ap_idx, ap_pos in zip(ap_idxs, ap_posns):
                ap = ap_array[ap_idx]
                others = [ap[k] for k in range(4) if k != ap_pos]
                
                others_match_old = 0
                others_match_new = 0
                for o in others:
                    c = replicas[r, o]
                    if c == old_color:
                        others_match_old += 1
                    else:
                        others_match_new += 1
                
                if others_match_old == 3:
                    delta -= 1
                if others_match_new == 3:
                    delta += 1
            
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                replicas[r, pos] = new_color
                replica_energies[r] += delta
                if replica_energies[r] < best_energy:
                    best_energy = int(replica_energies[r])
                    best_colors = replicas[r].copy()
            else:
                pass  # Reject, keep old state
        
        # Try replica exchange (swap)
        iter_count += 1
        if iter_count % swap_interval == 0:
            # Try to swap two adjacent replicas
            if rng.random() < 0.5:
                i = rng.randint(0, num_replicas - 2)
                j = i + 1
                # Metropolis criterion for swap
                dE = (replica_energies[i] - replica_energies[j]) * (1.0/temps[j] - 1.0/temps[i])
                if dE <= 0 or rng.random() < math.exp(-dE):
                    # Swap
                    replicas[[i, j]] = replicas[[j, i]]
                    replica_energies[[i, j]] = replica_energies[[j, i]]
        
        # If all replicas are stuck, perturb the coldest one
        if iter_count % 100 == 0:
            coldest = int(np.argmin(temps))  # This is always the last one
            # Actually find the coldest replica (highest temperature is index 0, lowest is last)
            # temps is geomspace from T_base to T_min, so index 0 is hottest, last is coldest
            coldest = num_replicas - 1
            num_flips = 5
            flip_positions = rng.choice(n, size=num_flips, replace=False)
            replicas[coldest, flip_positions] = 1 - replicas[coldest, flip_positions]
            replica_energies[coldest] = count_mono_vec(replicas[coldest])
            if replica_energies[coldest] < best_energy:
                best_energy = int(replica_energies[coldest])
                best_colors = replicas[coldest].copy()
    
    return best_colors.tolist()

# EVOLVE-BLOCK-END
