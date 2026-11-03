"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _generate_shell(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    # Number of such vectors: C(D, m) * 2^m
    # For D=11, m=4: 330 * 16 = 5280.
    # For D=11, m=2: 55 * 4 = 220.
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def _generate_seed_vectors():
    """Generate the initial set of vectors (same as seed)."""
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    return np.array(vectors, dtype=np.int64)

def construct():
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Strategy:
    # We want to maximize k such that for all distinct u, v in S:
    # ||u - v||^2 >= M^2, where M = max_{w in S} ||w||.
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # So we need ||u||^2 + ||v||^2 - 2 u.v >= M^2.
    
    # Let's consider a set S composed of vectors from shell m=4 and shell m=2.
    # Let S4 be a subset of shell 4, and S2 be a subset of shell 2.
    # M^2 = max(4, 4, 2, 2) = 4 (assuming S4 is non-empty).
    # 
    # Constraints:
    # 1. For u, v in S4: ||u||^2 + ||v||^2 - 2 u.v >= 4  => 4 + 4 - 2 u.v >= 4 => u.v <= 2.
    # 2. For u, v in S2: ||u||^2 + ||v||^2 - 2 u.v >= 4  => 2 + 2 - 2 u.v >= 4 => u.v <= 0.
    # 3. For u in S4, v in S2: ||u||^2 + ||v||^2 - 2 u.v >= 4  => 4 + 2 - 2 u.v >= 4 => u.v <= 1.
    
    # We will greedily select S4 first, then S2.
    # 
    # Step 1: Select S4 from shell 4 such that pairwise u.v <= 2.
    # Step 2: Select S2 from shell 2 such that:
    #          a) pairwise u.v <= 0 within S2
    #          b) u.v <= 1 for all u in S2, v in S4
    
    # Generate candidates
    shell4 = _generate_shell(4)
    shell2 = _generate_shell(2)
    
    n4 = len(shell4)
    n2 = len(shell2)
    
    # --- Select S4 ---
    # Greedy selection with vectorized updates
    idx4 = np.arange(n4)
    np.random.shuffle(idx4)
    shell4 = shell4[idx4]
    
    max_ip4 = np.full(n4, -np.inf)
    selected4_indices = []
    alive4 = np.ones(n4, dtype=bool)
    
    for i in range(n4):
        if not alive4[i]:
            continue
        if max_ip4[i] > 2:
            alive4[i] = False
            continue
            
        selected4_indices.append(i)
        alive4[i] = False
        
        # Update max_ip for all alive candidates
        dots = shell4 @ shell4[i]
        max_ip4 = np.maximum(max_ip4, dots)
        
        alive4 = max_ip4 <= 2
        if not np.any(alive4):
            break
            
    S4 = shell4[selected4_indices] if selected4_indices else np.empty((0, D), dtype=np.int64)
    
    # --- Select S2 ---
    # We need to filter shell2 based on S4 first.
    # Condition: for all v in S4, u.v <= 1.
    # Let max_ip_with_S4[u] = max_{v in S4} u.v. We need max_ip_with_S4[u] <= 1.
    
    if len(S4) > 0:
        # Compute max inner product of each shell2 vector with any S4 vector
        # S4 shape (k4, 11), shell2 shape (k2, 11)
        # Dots = shell2 @ S4.T -> shape (k2, k4)
        # max_ip_with_S4 = Dots.max(axis=1)
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        
        # Filter shell2 to only those with max_ip_24 <= 1
        valid_2_mask = max_ip_24 <= 1
        shell2_filtered = shell2[valid_2_mask]
    else:
        # If S4 is empty, M would be sqrt(2) if we only have S2.
        # Then constraint for S2 is u.v <= (2+2-2)/2 = 1? No.
        # If M^2 = 2, then 2+2-2u.v >= 2 => u.v <= 1.
        # But we want M to be as large as possible? No, M is determined by the set.
        # If S4 is empty, we just need a valid set from S2 with M^2=2.
        # Constraint: u.v <= 1.
        # However, our greedy for S2 below assumes constraint u.v <= 0 within S2 if M^2=4.
        # If S4 is empty, M^2=2, so constraint is u.v <= 1.
        # Let's handle the general case: determine M after selection?
        # Actually, if we pick S4, M^2=4. If S4 is empty, we should pick S2 with M^2=2.
        # But S4 greedy is very likely to find many vectors.
        # Let's assume S4 is non-empty for now, which implies M^2=4.
        # If S4 is empty, we fallback to a simpler logic or just return S2 with relaxed constraint.
        shell2_filtered = shell2
        
    n2f = len(shell2_filtered)
    
    # Now select from shell2_filtered such that pairwise u.v <= 0 (because M^2=4 from S4).
    # Wait, if S4 is empty, M^2=2, constraint is u.v <= 1.
    # Let's check if S4 is empty.
    if len(S4) == 0:
        # Use M^2 = 2 constraint: u.v <= 1
        max_ip_bound = 1
    else:
        # Use M^2 = 4 constraint: u.v <= 0
        max_ip_bound = 0
        
    idx2 = np.arange(n2f)
    np.random.shuffle(idx2)
    shell2_filtered = shell2_filtered[idx2]
    
    max_ip2 = np.full(n2f, -np.inf)
    selected2_indices = []
    alive2 = np.ones(n2f, dtype=bool)
    
    for i in range(n2f):
        if not alive2[i]:
            continue
        if max_ip2[i] > max_ip_bound:
            alive2[i] = False
            continue
            
        selected2_indices.append(i)
        alive2[i] = False
        
        dots = shell2_filtered @ shell2_filtered[i]
        max_ip2 = np.maximum(max_ip2, dots)
        
        alive2 = max_ip2 <= max_ip_bound
        if not np.any(alive2):
            break
            
    S2 = shell2_filtered[selected2_indices] if selected2_indices else np.empty((0, D), dtype=np.int64)
    
    # Combine S4 and S2
    if len(S4) > 0 and len(S2) > 0:
        result = np.vstack([S4, S2])
    elif len(S4) > 0:
        result = S4
    elif len(S2) > 0:
        result = S2
    else:
        # Fallback
        result = _generate_seed_vectors()
        
    # Final validation check to be safe (optional but good for debugging)
    # k = len(result)
    # if k > 1:
    #     norms_sq = np.sum(result**2, axis=1)
    #     M_sq = np.max(norms_sq)
    #     # Check min pairwise distance
    #     # This is O(k^2 * D), might be slow for large k, but k is small enough (~100-200)
    #     # Let's skip full validation in tight loop, trust the greedy.
    #     # But a quick sanity check on a sample?
    #     pass
        
    return result

# EVOLVE-BLOCK-END
