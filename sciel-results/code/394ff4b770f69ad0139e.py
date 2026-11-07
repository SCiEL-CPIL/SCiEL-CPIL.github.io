import time
import numpy as np
from collections import deque
from itertools import product

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Exploit the prism graph structure to reduce the search space.
    2. The signed adjacency matrix has a block structure where the spectrum
       depends on sign products along the two cycles and matching.
    3. Enumerate invariant sign patterns and find the minimum spectral radius
       configuration that guarantees a connected lift.
    4. Refine with local search on the full space.
    """
    
    deadline = time.time() + 85.0  # Safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge arrays for fast matrix construction
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Classify edges into top cycle, bottom cycle, and matching
    top_cycle_edges = []    # edges within 0..m-1
    bottom_cycle_edges = [] # edges within m..2m-1
    matching_edges = []     # edges between i and i+m
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # For each cycle, determine the edge indices in order
    # Top cycle: vertices 0,1,...,m-1
    # Bottom cycle: vertices m, m+1, ..., 2m-1
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        """
        f = [-1] * n
        f[0] = 0
        
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True
        
        return False

    def get_radius(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i in range(num_edges):
            val = float(signs[i])
            S[u_arr[i], v_arr[i]] = val
            S[v_arr[i], u_arr[i]] = val
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))
    
    # Key insight: The prism graph has a natural Z_m symmetry.
    # The signed adjacency matrix can be analyzed in terms of its action
    # on the Fourier modes of the cycle.
    #
    # For a cycle with signs s_0, s_1, ..., s_{m-1} (edges (0,1), (1,2), ..., (m-1,0)),
    # the product P = s_0 * s_1 * ... * s_{m-1} determines the "twist" of the cycle.
    #
    # The spectrum of the signed cycle is determined by the eigenvalues of the
    # signed adjacency matrix of the cycle, which for a cycle with twist P are:
    #   2*cos(2*pi*k/m) for P=1, or 2*cos((2*k+1)*pi/m) for P=-1
    #
    # For the prism graph, we have two cycles and a matching. The full spectrum
    # can be computed by considering the block structure.
    #
    # Let's define:
    #   - top_signs: signs of the m edges in the top cycle
    #   - bottom_signs: signs of the m edges in the bottom cycle  
    #   - match_signs: signs of the m matching edges
    #
    # The key observation is that the spectrum depends on:
    #   1. The product of signs around the top cycle: P_top = prod(top_signs)
    #   2. The product of signs around the bottom cycle: P_bot = prod(bottom_signs)
    #   3. The "relative twist" between the two cycles via the matching
    #
    # For the prism graph, the signed adjacency matrix has a specific structure.
    # Let's construct it and compute eigenvalues directly for each candidate pattern.
    
    # Strategy: Enumerate all 2^(2m) combinations of cycle sign products and matching patterns
    # For small m (4,5,7,9), this is feasible: 2^8=256, 2^10=1024, 2^14=16384, 2^18=262144
    
    # Actually, let's be smarter. The spectrum of the signed prism graph depends on
    # the sign patterns in a structured way. Let's use the fact that:
    # 
    # The signed adjacency matrix S of the prism graph can be written as a block matrix:
    # S = [[A_top, M], [M^T, A_bot]]
    # where A_top and A_bot are the signed adjacency matrices of the two cycles,
    # and M is the matching matrix with entries match_signs[i] at (i,i).
    #
    # For the eigenvalue problem, we can use the fact that the cycles have a natural
    # Fourier basis. Let's work in the Fourier basis.
    
    # For a cycle of length m, the eigenvectors are v_k[j] = exp(2*pi*i*k*j/m)
    # The signed adjacency matrix in the Fourier basis has a specific structure
    # depending on the sign pattern.
    
    # Let's take a different approach: directly enumerate sign patterns and compute
    # the spectral radius, but use the structure to reduce the search space.
    
    # The key insight from the inspiration: the spectrum is determined by sign products
    # of the two m-cycles and the matching. Let's enumerate:
    #   - For each of the 2^m possible sign patterns on the top cycle
    #   - For each of the 2^m possible sign patterns on the bottom cycle
    #   - For each of the 2^m possible sign patterns on the matching
    # This is 2^(3m) which is too large for m=9 (2^27).
    
    # Better approach: Use the fact that the prism graph has a symmetry.
    # The sign pattern can be characterized by:
    #   - The "phase" of the top cycle: how the signs rotate
    #   - The "phase" of the bottom cycle
    #   - The relative phase between top and bottom via the matching
    
    # Let's use a parameterization based on the sign products.
    # For a cycle, the product of all edge signs determines the "holonomy".
    # Let P_top = product of top cycle signs, P_bot = product of bottom cycle signs.
    # Let Q_i = top_sign[i] * match_sign[i] * bottom_sign[i] for each i (this is the
    # product around the "rectangle" formed by top edge i, matching i, and bottom edge i).
    # The product of all Q_i = P_top * P_bot * (product of match signs).
    
    # Actually, let's just do a smart enumeration. For m up to 9, we can:
    # 1. Fix the top cycle sign pattern (2^m possibilities)
    # 2. For each, try all bottom cycle patterns (2^m) and matching patterns (2^m)
    # But that's 2^(3m) which is too much.
    
    # Alternative: Use local search with a good initialization based on the structure.
    # Initialize with alternating signs on each cycle to create cancellations.
    
    def generate_structural_signs(m, num_edges, edges, top_cycle_edges, bottom_cycle_edges, matching_edges, 
                                   phase_top=0, phase_bot=0, phase_match=0):
        """Generate signs with alternating pattern on cycles."""
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Top cycle: alternate signs
        for idx in top_cycle_edges:
            u, v = edges[idx]
            pos = min(u, v)
            signs[idx] = 1 if ((pos + phase_top) % 2 == 0) else -1
        
        # Bottom cycle: alternate signs
        for idx in bottom_cycle_edges:
            u, v = edges[idx]
            pos = min(u - m, v - m)
            signs[idx] = 1 if ((pos + phase_bot) % 2 == 0) else -1
        
        # Matching: uniform sign
        for idx in matching_edges:
            signs[idx] = 1 if (phase_match == 0) else -1
        
        return signs
    
    best_signs = None
    best_radius = np.inf
    
    # Strategy 1: Try structural initializations with different phases
    for phase_top in [0, 1]:
        for phase_bot in [0, 1]:
            for phase_match in [0, 1]:
                signs = generate_structural_signs(m, num_edges, edges, top_cycle_edges, 
                                                   bottom_cycle_edges, matching_edges,
                                                   phase_top, phase_bot, phase_match)
                
                if not is_lift_connected(signs):
                    continue
                
                radius = get_radius(signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    # Strategy 2: Random initialization with local search
    rng = np.random.default_rng(3)
    
    # Try several random starts
    for trial in range(30):
        if time.time() > deadline:
            break
        
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        # Ensure connectivity
        attempts = 0
        while not is_lift_connected(signs) and attempts < 200:
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
            attempts += 1
        
        if not is_lift_connected(signs):
            continue
        
        radius = get_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Local search refinement with simulated annealing
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # First pass: greedy local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                if not is_lift_connected(current_signs):
                    current_signs[i] = old_sign
                    continue
                
                S = np.zeros((n, n), dtype=np.float64)
                for j in range(num_edges):
                    val = float(current_signs[j])
                    S[u_arr[j], v_arr[j]] = val
                    S[v_arr[j], u_arr[j]] = val
                
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
                
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
        
        # Second pass: simulated annealing for further improvement
        T = 0.05
        cooling_rate = 0.97
        steps_per_temp = 50
        
        sa_signs = best_signs.copy()
        sa_radius = best_radius
        
        while T > 0.005 and time.time() < deadline:
            for _ in range(steps_per_temp):
                if time.time() > deadline:
                    break
                
                idx = rng.integers(0, num_edges)
                old_sign = sa_signs[idx]
                sa_signs[idx] = -old_sign
                
                if not is_lift_connected(sa_signs):
                    sa_signs[idx] = old_sign
                    continue
                
                S = np.zeros((n, n), dtype=np.float64)
                for j in range(num_edges):
                    val = float(sa_signs[j])
                    S[u_arr[j], v_arr[j]] = val
                    S[v_arr[j], u_arr[j]] = val
                
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
                delta = new_radius - sa_radius
                
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    sa_radius = new_radius
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = sa_signs.copy()
                else:
                    sa_signs[idx] = old_sign
            
            T *= cooling_rate
    
    if best_signs is None:
        # Fallback: all +1 except one edge to ensure connectivity
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            best_signs[0] = -1
            if not is_lift_connected(best_signs):
                best_signs[1] = -1
    
    return best_signs
