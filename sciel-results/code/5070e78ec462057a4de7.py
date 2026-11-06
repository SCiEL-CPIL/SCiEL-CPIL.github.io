import numpy as np
import time
from itertools import combinations

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 55

    # 1. Generate the 256-word linear base code
    # Generator matrix for [17,9,6] shortened QR code
    G = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    ], dtype=np.uint8)

    # Generate codewords
    base_code = []
    for i in range(2**9):
        bits = [(i >> j) & 1 for j in range(9)]
        cw = np.dot(bits, G) % 2
        val = 0
        for b in cw:
            val = (val << 1) | b
        base_code.append(val)
        
    # Verify min distance of base code
    min_d = N
    for i in range(len(base_code)):
        for j in range(i+1, len(base_code)):
            d = bin(base_code[i] ^ base_code[j]).count('1')
            if d < min_d:
                min_d = d
            if min_d < MIN_DIST:
                break
        if min_d < MIN_DIST:
            break

    if min_d < MIN_DIST:
        # Fallback to greedy if hardcoded matrix is invalid
        # ... (insert greedy code here) ...
        # For now, return empty or greedy result
        # To keep it simple and safe, if matrix fails, use greedy
        return _greedy_construct(N, MIN_DIST, start_time, time_limit)

    # 2. Prepare for extension search
    # Compute Parity Check Matrix H (8x17)
    # H = [P^T | I] if G = [I | P]
    P = G[:, 9:]
    H = np.hstack((P.T, np.eye(8, dtype=np.uint8)))
    
    # Precompute min weight for each syndrome
    # syndrome is 8 bits.
    min_weight_for_syndrome = [100] * 256
    # Iterate through all 2^17 vectors to find min weight for each syndrome
    # This is 131072 iterations.
    for v in range(1 << N):
        # Compute syndrome
        # s = H @ v_vec
        # Optimization: use bit operations or numpy
        # Convert v to vector
        # v_vec = np.array([(v >> i) & 1 for i in range(N)], dtype=np.uint8)
        # s = H @ v_vec % 2
        # This is slow in Python loop.
        
        # Alternative: Just check candidates directly?
        # Candidates are weight 8 and 9.
        # We can just compute distance to base code for candidates directly?
        # No, syndrome method is better if we can compute it fast.
        
        # Let's use a faster syndrome computation.
        # H is 8x17.
        # s_i = sum(H[i,j] * v_j) mod 2
        pass

    # Simpler approach for extension:
    # Candidates: weight 8 and 9 vectors.
    candidates = []
    # Generate weight 8 and 9 masks
    for d in [8, 9]:
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            candidates.append(mask)
            
    # Filter candidates: must be dist >= 6 from ALL base codewords
    valid_candidates = []
    for c in candidates:
        ok = True
        for b in base_code:
            if bin(c ^ b).count('1') < MIN_DIST:
                ok = False
                break
        if ok:
            valid_candidates.append(c)
            
    # If no valid candidates, return base code
    if not valid_candidates:
        return base_code

    # 3. Find maximum clique in compatibility graph of valid_candidates
    # Graph: edge if dist >= 6
    # Use greedy heuristic
    n_cand = len(valid_candidates)
    if n_cand == 0:
        return base_code
        
    # Build adjacency matrix or list
    adj = [[False] * n_cand for _ in range(n_cand)]
    for i in range(n_cand):
        for j in range(i+1, n_cand):
            if bin(valid_candidates[i] ^ valid_candidates[j]).count('1') >= MIN_DIST:
                adj[i][j] = True
                adj[j][i] = True
                
    # Greedy clique finding
    clique = []
    available = list(range(n_cand))
    
    # Simple greedy: pick node with highest degree, add to clique, remove incompatible
    # Repeat
    while available:
        # Find node with max degree in available subgraph
        max_deg = -1
        max_node = -1
        for u in available:
            deg = sum(1 for v in available if adj[u][v])
            if deg > max_deg:
                max_deg = deg
                max_node = u
                
        clique.append(max_node)
        # Remove max_node and its non-neighbors from available
        new_available = []
        for u in available:
            if u != max_node and adj[u][max_node]:
                new_available.append(u)
        available = new_available
        
    extended_code = base_code + [valid_candidates[i] for i in clique]
    return extended_code

def _greedy_construct(N, MIN_DIST, start_time, time_limit):
    # Fallback greedy implementation
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
            
    blocked = bytearray(1 << N)
    code = []
    
    def add_word(w):
        if blocked[w]:
            return
        code.append(w)
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1
                
    add_word(0)
    for w in range(1, 1 << N):
        if time.time() - start_time > time_limit:
            break
        if not blocked[w]:
            add_word(w)
            
    return code
