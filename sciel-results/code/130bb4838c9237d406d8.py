import time
import math
from collections import defaultdict

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    # Petersen Graph as base
    # Vertices 0-9
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    num_edges = len(edges)
    
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Generate all closed non-backtracking walks of length < g from every vertex
    # We store them as lists of (edge_id, sign)
    walks = []
    for start_v in range(n_base):
        # BFS/DFS for non-backtracking walks
        # State: (current_v, prev_edge, length, path)
        # We use a list as a stack for DFS to avoid queue overhead
        stack = []
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            stack.append((next_v, e_id, 1, [(e_id, sign)]))
        
        while stack:
            curr_v, prev_e, length, path = stack.pop()
            if length >= g:
                continue
            if curr_v == start_v and length >= 3:
                walks.append(tuple(path))
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                stack.append((next_v, e_id, length + 1, new_path))
    
    # Deduplicate walks
    unique_walks = list(set(walks))
    
    # Pre-compute coefficient vectors for each walk
    # constraint: sum(c_i * v_i) != 0 mod m
    constraints = []
    for w in unique_walks:
        coeffs = [0] * num_edges
        for e_id, sign in w:
            coeffs[e_id] += sign
        # Store only non-zero coefficients
        nonzero = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        if nonzero:
            constraints.append(nonzero)
    
    # Sort constraints by number of variables (tightest first) to prune earlier?
    # Actually, for backtracking, we want to check constraints as soon as all their vars are assigned.
    # Let's group constraints by the max index of their variables.
    # If a constraint only involves vars 0..k, we can check it once var k is assigned.
    constraints_by_max_var = [[] for _ in range(num_edges)]
    for c in constraints:
        max_var = max(idx for idx, _ in c)
        constraints_by_max_var[max_var].append(c)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    best_volts = None
    
    # Moore bound for n(3,16) is 512. n = 10*m. So m >= 52.
    # Try m values.
    # The smallest known is 936 (m=93.6? No, 936 is not a multiple of 10. 
    # Wait, the Petersen lift has n=10m. 
    # If the best known is 936, maybe it's not a Petersen lift? 
    # But we are forced to use the Petersen lift structure in this code.
    # The prompt says "Keep the function names...". It doesn't force the *structure* 
    # but the current code uses Petersen. 
    # However, the inspiration says "Voltage lifting of the Petersen graph... significantly smaller...".
    # Let's stick to Petersen lift but optimize the search.
    
    # We should probably try m values that are likely to work.
    # For girth 16, m needs to be large enough to avoid short cycles.
    # Heuristic: m should be > g/2? No, much larger.
    # Let's start from m=52 and go up.
    
    for m in range(52, 2000):
        if time.time() > deadline - 10.0:
            break
        
        # Backtracking search for voltages v[0..14] in 1..m-1
        # v[i] != 0
        vols = [0] * num_edges
        found = False
        
        # Order of assignment: 0, 1, ..., 14
        # We can try to optimize variable ordering, but fixed is fine for now.
        
        def backtrack(idx):
            nonlocal found
            if time.time() > deadline - 5.0:
                found = "timeout"
                return
            
            if idx == num_edges:
                found = True
                return
            
            # Try all possible voltages for edge idx
            # Optimization: If m is large, 1..m-1 is a lot.
            # But constraints are modulo m.
            # We can prune if a constraint involving only prev vars is already violated?
            # No, we check constraints when the max var is assigned.
            
            # Check constraints that become "complete" at this step?
            # No, we check them when the *last* variable in the constraint is assigned.
            # So at step idx, we assign v[idx], then check all constraints where max_var == idx.
            
            # Heuristic: Try values that are "independent" or just 1, 2, ...
            # To speed up, we can try a random subset or specific patterns if m is huge.
            # But m is likely < 1000. 1000^15 is impossible.
            # Pruning is key.
            
            # Let's try values 1 to m-1.
            # To avoid symmetry issues (v and -v might be equivalent), we could fix v[0]=1?
            # No, voltages are not just signs, they are values.
            # However, scaling all voltages by a unit u in Z_m^* preserves the property 
            # that sum != 0. So we can fix v[0] = 1 (if m is prime or 1 is a unit).
            # If m is not prime, 1 is always a unit.
            # So we can fix v[0] = 1 to reduce search space by factor m.
            
            if idx == 0:
                candidates = [1]
            else:
                candidates = range(1, m)
                
            # Sort candidates? 
            # Maybe try values that are coprime to m first?
            # Or just iterate.
            
            for val in candidates:
                vols[idx] = val
                
                # Check constraints where max_var == idx
                valid = True
                for c in constraints_by_max_var[idx]:
                    s = 0
                    for i, coeff in c:
                        s += coeff * vols[i]
                    if s % m == 0:
                        valid = False
                        break
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                if found == "timeout":
                    return False
            
            return False
        
        if backtrack(0):
            if found:
                best_m = m
                best_n = 10 * m
                best_volts = vols[:]
                break
            elif found == "timeout":
                break
    
    if best_edges is None and best_volts is not None:
        lift_edges = []
        for i, (u, v) in enumerate(edges):
            vol = best_volts[i]
            for t in range(best_m):
                u_lift = u * best_m + t
                v_lift = v * best_m + (t + vol) % best_m
                lift_edges.append((u_lift, v_lift))
        best_edges = lift_edges
    
    if best_edges is None:
        return _luw_fallback(k, g)
    
    return best_n, best_edges

def _luw_fallback(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
