import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an edge colouring for K16.
    Uses a local search to find a colouring with minimal colours (target 13-15).
    Falls back to the cyclic 17-colour construction if time runs out or no solution found.
    """
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - (i * (i + 1) // 2) + (j - i - 1)
        
    # Precompute K4 edge tuples
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4_edges.append(edges)
    
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)
    num_k4s = len(k4_edges)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    def is_valid(colors):
        for edges in k4_edges:
            if len(set(colors[e] for e in edges)) < 5:
                return False
        return True

    def cost(colors):
        total = 0
        for edges in k4_edges:
            c = [colors[e] for e in edges]
            d = len(set(c))
            if d < 5:
                total += (5 - d) ** 2
        return total

    def fallback():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    def local_search(k, time_limit):
        start_time = time.time()
        best_result = None
        best_cost_found = float('inf')
        
        # We will do multiple restarts within the time limit
        while time.time() - start_time < time_limit:
            # Random init
            colors = np.random.randint(0, k, size=num_edges, dtype=np.int32)
            cur_cost = cost(colors)
            if cur_cost == 0:
                return colors.tolist()
            
            if cur_cost < best_cost_found:
                best_cost_found = cur_cost
                best_result = colors.copy()
                
            # Local search loop
            no_improve_count = 0
            max_no_improve = 1000 # Reset after this many non-improving steps
            
            while time.time() - start_time < time_limit:
                # Pick a random K4 that is violating
                # To be efficient, we could maintain a list, but for now random pick
                # If we pick a valid one, we just skip or pick a random edge globally?
                # Better: pick a random edge globally? No, target violations.
                
                # Optimization: Maintain a list of violating K4s?
                # For now, just pick random K4. If valid, try again or break inner loop?
                # If most K4s are valid, this is slow.
                # Let's pick a random edge and try to improve it?
                # Or pick a random K4, if valid, pick another.
                
                # Heuristic: Pick a random K4.
                k4_idx = random.randint(0, num_k4s - 1)
                edges = k4_edges[k4_idx]
                c = [colors[e] for e in edges]
                d = len(set(c))
                
                if d >= 5:
                    # This K4 is fine. 
                    # To avoid wasting time, we can just pick a random edge globally and try to improve.
                    # Or just continue searching for a violating K4.
                    # Given the density of constraints, likely many are violated initially.
                    # As we progress, fewer violations.
                    # If we hit too many valid K4s, it's likely we are close to a solution or stuck.
                    # Let's just pick a random edge globally in this case.
                    e_idx = random.randint(0, num_edges - 1)
                else:
                    # Pick a random edge from this violating K4
                    e_idx = edges[random.randint(0, 5)]
                
                old_c = colors[e_idx]
                
                # Try all colours for this edge
                best_c = old_c
                best_c_cost = cur_cost
                
                # Optimization: Only recalculate cost for affected K4s?
                # The cost function is global.
                # But we can calculate delta cost.
                # delta = new_cost - old_cost
                # new_cost = old_cost - sum(old_penalties) + sum(new_penalties)
                
                # Let's implement delta cost for speed.
                # We need to know the current penalty of each K4.
                # This requires maintaining an array k4_penalties.
                # Let's do that outside the loop.
                
                # For now, full cost is ~1820 * small ops.
                # If k=15, 14 iterations per edge.
                # 1820 * 14 * 1000 iters = 25M ops. Might be okay in 1s?
                # Python is slow. 25M simple ops ~ 1-2s.
                # We have 30s budget.
                
                # Let's try to optimize cost calculation.
                # Use a helper that computes cost given colors array.
                
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    colors[e_idx] = new_c
                    new_cost = cost(colors)
                    if new_cost < best_c_cost:
                        best_c_cost = new_cost
                        best_c = new_c
                
                colors[e_idx] = best_c
                
                if best_c != old_c:
                    cur_cost = best_c_cost
                    no_improve_count = 0
                    if cur_cost == 0:
                        return colors.tolist()
                    if cur_cost < best_cost_found:
                        best_cost_found = cur_cost
                        best_result = colors.copy()
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break # Restart
                
        if best_result is not None and best_cost_found == 0:
            return best_result.tolist()
            
        return None

    # Try to find solutions with 13, 14, 15 colours
    time_start = time.time()
    time_limit_per_k = 30.0
    
    for k in [13, 14, 15]:
        if time.time() - time_start > 100:
            break
        res = local_search(k, time_limit_per_k)
        if res is not None:
            if is_valid(res):
                return res
    
    return fallback()
# EVOLVE-BLOCK-END
