import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # Safety margin for 150s limit

def _score_from_bitsets(adj_bits, deg):
    """
    Calculate the maximum degree sum over all triangles using bitsets.
    adj_bits: list of integers, adj_bits[i] has bit j set if edge (i,j) exists.
    deg: list of degrees.
    """
    best = 0
    for i in range(N):
        # Neighbors of i
        ni = adj_bits[i]
        # Iterate over neighbors j of i with j > i to avoid double counting
        # We can just iterate all neighbors and check j > i
        j = 0
        # Extract neighbors
        while ni:
            if ni & 1:
                # j is the current bit position
                if j > i:
                    # Check triangles (i, j, k) where k is neighbor of both i and j
                    # Common neighbors of i and j
                    common = ni & adj_bits[j]
                    # k must be > j to ensure unique triangle (i, j, k) with i<j<k? 
                    # Actually, for i<j, k can be anything. 
                    # To be efficient, let's just iterate k in common.
                    # We need max deg[i]+deg[j]+deg[k].
                    # deg[i] and deg[j] are fixed for this pair.
                    # We want max deg[k] among k in common.
                    # Since N=16, we can just iterate.
                    max_k_deg = -1
                    c = common
                    k = 0
                    while c:
                        if c & 1:
                            if deg[k] > max_k_deg:
                                max_k_deg = deg[k]
                        c >>= 1
                        k += 1
                    if max_k_deg >= 0:
                        s = deg[i] + deg[j] + max_k_deg
                        if s > best:
                            best = s
            ni >>= 1
            j += 1
    return best

def _update_deg(adj_bits, deg, u, v, add):
    """Update degrees and adjacency bits for adding/removing edge (u,v)."""
    if add:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
    else:
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    best_score = float('inf')
    best_adj_bits = None
    best_deg = None
    
    # Precompute all possible edges and non-edges indices for random selection
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Initial construction: K_{8,8} + 1 edge
    # This is a good starting point because it's close to the extremal graph.
    
    num_restarts = 50 # We can afford more restarts now
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE:
            break
            
        # Initialize adj_bits
        adj_bits = [0] * N
        deg = [0] * N
        
        # Add K_{8,8} edges
        for i in range(8):
            for j in range(8, 16):
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                deg[i] += 1
                deg[j] += 1
        
        # Add 1 random internal edge to make it 65 edges
        # K_{8,8} has 64 edges.
        # We need to add 1 edge within part A (0-7) or part B (8-15).
        part = rng.choice([range(8), range(8, 16)])
        u, v = rng.sample(list(part), 2)
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
        cur_score = _score_from_bitsets(adj_bits, deg)
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj_bits = adj_bits[:]
            best_deg = deg[:]
            
        # Simulated Annealing
        # Parameters
        start_temp = 2.0
        end_temp = 0.01
        # Estimate iterations per restart based on time
        # Let's aim for ~2 seconds per restart
        # 50 restarts * 2s = 100s.
        # We can run until deadline.
        
        # We need a list of current edges and non-edges for random selection
        # But maintaining lists is slow. 
        # Alternative: Pick random pairs (u,v) and (x,y) from all_pairs.
        # Check if (u,v) is edge and (x,y) is non-edge.
        # Probability of picking valid edge is 65/120 ~ 0.54.
        # Probability of picking valid non-edge is 55/120 ~ 0.46.
        # Combined prob ~ 0.25. So 4 tries on average.
        # This is acceptable.
        
        max_iters = 20000 # High iteration count
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE:
                break
                
            # Pick random swap
            # To improve efficiency, we can bias towards edges incident to high-degree vertices
            # but for now, let's stick to random with validity check.
            
            attempts = 0
            valid_swap = False
            while attempts < 20: # Max attempts to find valid swap
                u, v = all_pairs[rng.randrange(len(all_pairs))]
                x, y = all_pairs[rng.randrange(len(all_pairs))]
                
                # Check (u,v) is edge
                if not (adj_bits[u] & (1 << v)):
                    attempts += 1
                    continue
                # Check (x,y) is non-edge
                if adj_bits[x] & (1 << y):
                    attempts += 1
                    continue
                # Ensure they are not the same edge (swapping edge with itself is no-op)
                if (u == x and v == y) or (u == y and v == x):
                    attempts += 1
                    continue
                valid_swap = True
                break
                
            if not valid_swap:
                continue
                
            # Apply swap
            _update_deg(adj_bits, deg, u, v, False)
            _update_deg(adj_bits, deg, x, y, True)
            
            new_score = _score_from_bitsets(adj_bits, deg)
            
            # Acceptance
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.5
            else:
                delta = new_score - cur_score
                if delta > 5: # Reject large jumps
                    accept = False
                elif temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > rng.random()
            
            if accept:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj_bits = adj_bits[:]
                    best_deg = deg[:]
            else:
                # Undo swap
                _update_deg(adj_bits, deg, x, y, False)
                _update_deg(adj_bits, deg, u, v, True)
                
            temp *= cooling_rate

    # Convert best_adj_bits to edge list
    if best_adj_bits is None:
        # Fallback
        adj_bits = [0] * N
        for i in range(8):
            for j in range(8, 16):
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
        adj_bits[0] |= (1 << 1)
        adj_bits[1] |= (1 << 0)
        best_adj_bits = adj_bits

    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj_bits[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': N, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(len(res['edges']))
