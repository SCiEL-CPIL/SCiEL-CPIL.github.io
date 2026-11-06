"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

# Increase recursion limit for branch and bound
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline to be safe within 90s limit
    
    N = 256
    nodes = list(range(1, N + 1))
    
    # Weights are reciprocals
    weights = [1.0 / x for x in nodes]
    
    # Build conflict graph: edge (u, v) exists if u and v form a 3-AP with some w in [1, N]
    # u, w, v are in AP if w - u = v - w => 2w = u + v => u + v is even and w = (u+v)/2 is in [1, N]
    # Or u, v, w are in AP if v - u = w - v => 2v = u + w => w = 2v - u is in [1, N]
    # Or w, u, v are in AP if u - w = v - u => 2u = w + v => w = 2u - v is in [1, N]
    
    # Actually, the condition "no 3-term AP" means for any distinct a, b, c in S, 
    # it is not the case that a, b, c form an AP.
    # This is equivalent to: for any pair (u, v) in S, there is no w in S such that 
    # (u, w, v) or (u, v, w) or (w, u, v) is an AP.
    # 
    # Let's define the conflict graph G where vertices are 1..N.
    # Two vertices u and v are connected by an edge if they cannot both be in S.
    # They cannot both be in S if there exists a w in [1, N] such that {u, v, w} contains a 3-AP.
    # The 3-APs involving u and v are:
    # 1. u, w, v => w = (u+v)/2. If u+v is even and 1 <= (u+v)/2 <= N, then u and v conflict.
    # 2. u, v, w => w = 2v - u. If 1 <= 2v - u <= N and w != u and w != v, then u and v conflict.
    # 3. w, u, v => w = 2u - v. If 1 <= 2u - v <= N and w != u and w != v, then u and v conflict.
    
    # Note: if u=v, it's not a pair. We consider distinct u, v.
    # If w == u or w == v, it's a constant progression or not 3 distinct terms?
    # The problem says "nonconstant k=3 term arithmetic progression".
    # So a, b, c must be distinct.
    # If w = (u+v)/2 and w != u, w != v (which implies u != v), then it's a valid conflict.
    # If w = 2v - u and w != u, w != v (which implies u != v), then it's a valid conflict.
    # If w = 2u - v and w != u, w != v (which implies u != v), then it's a valid conflict.
    
    # Let's build the adjacency matrix or list.
    # N=256 is small enough for a 256x256 boolean matrix or bitset.
    # Using a list of bitsets (integers) for fast operations.
    
    conflict = [0] * (N + 1)
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # Check 1: w = (u+v)/2
            s = u + v
            if s % 2 == 0:
                w = s // 2
                if 1 <= w <= N and w != u and w != v:
                    # Conflict
                    conflict[u] |= (1 << v)
                    conflict[v] |= (1 << u)
                    continue # No need to check other cases? 
                    # Actually, if they conflict for one reason, they conflict.
                    # But let's be sure. If they conflict, we just need to know it.
            
            # Check 2: w = 2v - u
            w = 2 * v - u
            if 1 <= w <= N and w != u and w != v:
                conflict[u] |= (1 << v)
                conflict[v] |= (1 << u)
                continue
            
            # Check 3: w = 2u - v
            w = 2 * u - v
            if 1 <= w <= N and w != u and w != v:
                conflict[u] |= (1 << v)
                conflict[v] |= (1 << u)
    
    # MWIS Branch and Bound
    # State: (available_nodes, current_weight, current_set)
    # available_nodes: bitset of nodes that can still be added
    # We order nodes by weight (descending) to prune better? 
    # Standard MWIS B&B:
    # 1. Order vertices by weight descending.
    # 2. At each step, pick a vertex v.
    #    Branch 1: Include v. Remove v and its neighbors from available. Add weight[v].
    #    Branch 2: Exclude v. Remove v from available.
    # 3. Upper bound: current_weight + sum of weights of all remaining available nodes.
    #    (This is a valid upper bound, though loose. Greedy coloring bound is better but complex.)
    #    For N=256, simple sum bound might be too slow if not pruned well.
    #    However, the graph is sparse? 
    #    Let's check density.
    #    Each node conflicts with roughly O(N) nodes?
    #    For a fixed u, v conflicts if (u+v)/2 is int in range, or 2v-u in range, or 2u-v in range.
    #    This is roughly 3/4 of all other nodes? No, let's estimate.
    #    For random u, v, (u+v)/2 is in [1, N] with prob ~1. 2v-u in [1, N] with prob ~1/2. 2u-v in [1, N] with prob ~1/2.
    #    So density is high.
    #    
    #    Alternative: Use a heuristic to get a good initial solution, then B&B to prove optimality or find better.
    #    Given the time limit, we might not find the *exact* optimal if the search space is huge, 
    #    but B&B with a good bound will likely find a very good solution quickly.
    
    # Let's use a simple recursive B&B with the "sum of remaining weights" bound.
    # To make it faster, we can use a bitset for available nodes.
    
    # Sort nodes by weight descending
    sorted_nodes = sorted(range(1, N + 1), key=lambda x: -weights[x-1])
    
    # Map node index to position in sorted list? 
    # Actually, we can just iterate through the sorted list.
    # State: index in sorted list, available bitset, current_weight, current_set (list)
    
    best_weight = 0.0
    best_set = []
    
    # Initial greedy solution to set a good lower bound
    # Greedy: pick highest weight node, remove it and its conflicts, repeat.
    avail = (1 << (N + 1)) - 2  # bits 1..N set
    greedy_set = []
    greedy_weight = 0.0
    for u in sorted_nodes:
        if avail & (1 << u):
            greedy_set.append(u)
            greedy_weight += weights[u-1]
            # Remove u and its neighbors
            avail &= ~(1 << u)
            avail &= ~conflict[u]
    
    best_weight = greedy_weight
    best_set = list(greedy_set)
    
    def bb(idx, avail, curr_w, curr_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
            
        if idx == N:
            if curr_w > best_weight:
                best_weight = curr_w
                best_set = list(curr_set)
            return
            
        # Upper bound: curr_w + sum(weights of all available nodes)
        # Precompute prefix sums of weights? 
        # The available set changes, so we need sum of weights of nodes in `avail`.
        # Calculating this every time is O(N). 
        # Optimization: pass the sum of available weights as a parameter?
        # When we remove a node, we subtract its weight.
        
        # Let's modify bb to take `avail_sum`
        pass
        
    # Let's restructure with avail_sum
    
    def bb(idx, avail, curr_w, curr_set, avail_sum):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
            
        # Prune if upper bound is not better
        if curr_w + avail_sum <= best_weight:
            return
            
        if idx == N:
            if curr_w > best_weight:
                best_weight = curr_w
                best_set = list(curr_set)
            return
            
        u = sorted_nodes[idx]
        
        # Option 1: Include u
        if avail & (1 << u):
            new_avail = avail & ~(1 << u) & ~conflict[u]
            # Calculate new avail_sum
            # We need to know which nodes were removed.
            # Removed = (1 << u) | conflict[u]
            # But we only remove if they are in avail.
            # removed_mask = (1 << u) | conflict[u]
            # actual_removed = new_avail ^ avail  # No, new_avail = avail & ~removed_mask
            # actual_removed = avail & ((1 << u) | conflict[u])
            
            removed_mask = (1 << u) | conflict[u]
            actual_removed = avail & removed_mask
            
            # Sum of weights of actual_removed
            # This is O(N) if we iterate. 
            # To speed up, we can precompute weight sums for bitsets? No, too many.
            # Since N=256, iterating through bits is fast in Python if done carefully? 
            # Or just calculate avail_sum - sum(weights of removed)
            
            # Let's just calculate the sum of weights of nodes in actual_removed
            rem_sum = 0.0
            m = actual_removed
            while m:
                lsb = m & -m
                node_idx = lsb.bit_length() - 1
                rem_sum += weights[node_idx - 1]
                m ^= lsb
                
            new_avail_sum = avail_sum - rem_sum
            
            curr_set.append(u)
            bb(idx + 1, new_avail, curr_w + weights[u-1], curr_set, new_avail_sum)
            curr_set.pop()
            
        # Option 2: Exclude u
        # new_avail = avail & ~(1 << u)
        # new_avail_sum = avail_sum - weights[u-1] (if u was in avail)
        
        if avail & (1 << u):
            new_avail = avail & ~(1 << u)
            new_avail_sum = avail_sum - weights[u-1]
            bb(idx + 1, new_avail, curr_w, curr_set, new_avail_sum)
        else:
            # If u is not available, we just move to next
            bb(idx + 1, avail, curr_w, curr_set, avail_sum)
            
    # Initial avail_sum
    initial_avail = (1 << (N + 1)) - 2
    initial_avail_sum = sum(weights)
    
    # Start B&B
    # To improve pruning, we can process nodes in order of decreasing weight.
    # The `sorted_nodes` list is already sorted by weight descending.
    
    bb(0, initial_avail, 0.0, [], initial_avail_sum)
    
    return best_set

# EVOLVE-BLOCK-END
