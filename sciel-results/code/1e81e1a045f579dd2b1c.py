"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 256

    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Build conflict graph: edge between u and v if gcd(u,v) > 1
    # We want Maximum Weight Independent Set on conflict graph
    # = Maximum Weight Clique on complement graph (coprime graph)
    
    n_nodes = N  # Nodes 0..255 correspond to values 1..256
    weights = [i + 1 for i in range(n_nodes)]
    
    # Build coprime adjacency as bitmasks
    adj_mask = [0] * n_nodes
    for i in range(n_nodes):
        vi = i + 1
        for j in range(i + 1, n_nodes):
            vj = j + 1
            if math.gcd(vi, vj) == 1:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    
    # Precompute weight sums for byte-sized chunks for fast mask weight computation
    # For each byte position b (0..31) and each 8-bit pattern p (0..255),
    # precompute the sum of weights of nodes in positions [8*b, 8*b+7] that are set in p.
    byte_sums = [[0] * 256 for _ in range(32)]
    for b in range(32):
        base = b * 8
        for p in range(256):
            s = 0
            for k in range(8):
                if p & (1 << k):
                    node = base + k
                    if node < n_nodes:
                        s += weights[node]
            byte_sums[b][p] = s

    def mask_weight_sum(mask):
        """Compute sum of weights for nodes in mask using precomputed byte sums."""
        total = 0
        for b in range(32):
            byte_val = (mask >> (b * 8)) & 0xFF
            if byte_val:
                total += byte_sums[b][byte_val]
        return total

    # Greedy heuristic to get initial lower bound
    def greedy_clique():
        P = (1 << n_nodes) - 1
        R = 0
        w = 0
        while P:
            # Pick max weight/degree ratio node in P for better heuristic
            best_node = -1
            best_ratio = -1
            pm = P
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                deg = (P & adj_mask[idx]).bit_count()
                ratio = weights[idx] / max(deg, 1)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_node = idx
                pm ^= lsb
            
            if best_node == -1:
                break
            
            R |= (1 << best_node)
            w += weights[best_node]
            P = P & adj_mask[best_node] & ~(1 << best_node)
        
        return R, w

    init_R, init_w = greedy_clique()
    global_best_weight = init_w
    global_best_mask = init_R

    # Branch and bound with Bron-Kerbosch style pivot
    sys.setrecursionlimit(10000)
    
    def bk_search(R_mask, P_mask, X_mask, current_weight):
        nonlocal global_best_weight, global_best_mask
        
        if time.time() > deadline:
            return
            
        if not P_mask:
            if current_weight > global_best_weight:
                global_best_weight = current_weight
                global_best_mask = R_mask
            return
            
        # Upper bound: current_weight + sum of all weights in P
        # Quick check: if even adding all of P can't beat best, prune
        bound = current_weight + mask_weight_sum(P_mask)
        if bound <= global_best_weight:
            return
            
        # Pivot selection: pick node in P with max degree in P
        pivot_idx = -1
        max_deg = -1
        pm = P_mask
        while pm:
            lsb = pm & -pm
            idx = lsb.bit_length() - 1
            deg = (P_mask & adj_mask[idx]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot_idx = idx
            pm ^= lsb
        
        if pivot_idx == -1:
            lsb = P_mask & -P_mask
            pivot_idx = lsb.bit_length() - 1
        
        # Branch on vertices in P that are NOT adjacent to pivot
        # These are the candidates that must be considered (vertices adjacent to pivot 
        # are covered by the pivot's recursive calls in standard BK logic)
        verts_to_iter = P_mask & ~adj_mask[pivot_idx]
        
        # Collect and sort by weight descending for better pruning
        verts_list = []
        vm = verts_to_iter
        while vm:
            lsb = vm & -vm
            idx = lsb.bit_length() - 1
            verts_list.append(idx)
            vm ^= lsb
        
        verts_list.sort(key=lambda x: weights[x], reverse=True)
        
        for v in verts_list:
            if time.time() > deadline:
                return
            
            new_R = R_mask | (1 << v)
            new_P = P_mask & adj_mask[v] & ~(1 << v)
            new_X = X_mask & adj_mask[v] & ~(1 << v)
            new_weight = current_weight + weights[v]
            
            bk_search(new_R, new_P, new_X, new_weight)
            
            # Update P and X for next iteration
            P_mask &= ~(1 << v)
            X_mask |= (1 << v)
    
    # Initial call
    initial_P = (1 << n_nodes) - 1
    bk_search(0, initial_P, 0, 0)
    
    # Local search polish: try to add nodes or swap nodes to improve
    def try_improve(result_mask, result_weight):
        """Try to add any node that's coprime to all in result, or swap pairs."""
        nonlocal global_best_weight, global_best_mask
        improved = True
        while improved:
            improved = False
            # Try adding a single node
            for v in range(n_nodes):
                if result_mask & (1 << v):
                    continue
                # Check if v is coprime to all nodes in result
                compatible = True
                rm = result_mask
                while rm:
                    lsb = rm & -rm
                    u = lsb.bit_length() - 1
                    if not (result_mask & adj_mask[v]) or v == u:
                        pass
                    if (adj_mask[v] & (1 << u)) == 0:
                        compatible = False
                        break
                    rm ^= lsb
                if compatible:
                    result_mask |= (1 << v)
                    result_weight += weights[v]
                    improved = True
                    break
            # Try swapping: remove one node, add a better one
            if not improved:
                rm = result_mask
                while rm:
                    lsb = rm & -rm
                    u = lsb.bit_length() - 1
                    # Remove u, try to add a better node
                    new_mask = result_mask & ~(1 << u)
                    for v in range(n_nodes):
                        if new_mask & (1 << v):
                            continue
                        if v == u:
                            continue
                        if weights[v] <= weights[u]:
                            continue
                        # Check if v is coprime to all in new_mask
                        compatible = True
                        nm = new_mask
                        while nm:
                            lsb2 = nm & -nm
                            w2 = lsb2.bit_length() - 1
                            if (adj_mask[v] & (1 << w2)) == 0:
                                compatible = False
                                break
                            nm ^= lsb2
                        if compatible:
                            new_mask |= (1 << v)
                            new_weight = result_weight - weights[u] + weights[v]
                            if new_weight > result_weight:
                                result_mask = new_mask
                                result_weight = new_weight
                                improved = True
                                break
                    if improved:
                        break
                    rm ^= lsb
        return result_mask, result_weight

    # Apply polish
    polished_mask, polished_weight = try_improve(global_best_mask, global_best_weight)
    if polished_weight > global_best_weight:
        global_best_weight = polished_weight
        global_best_mask = polished_mask

    # Extract result
    result = []
    mask = global_best_mask
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        mask ^= lsb
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # Fallback: return 1 and all primes (valid but suboptimal)
                return [1] + primes
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
