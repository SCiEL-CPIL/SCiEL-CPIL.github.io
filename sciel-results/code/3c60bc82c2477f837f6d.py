import time
import sys
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0

    if k != 4 or g != 10:
        return _construct_luw(k, g, deadline)

    MOD = 40
    N_A = 4
    N_B = 4
    NUM_VARS = 16
    
    # Precompute all closed walks of length 4, 6, 8
    # Walk is a list of (var_idx, sign)
    # var_idx = i*4 + j for voltage v[i][j]
    
    def get_closed_walks():
        walks = []
        # Length 4: a0-b1-a1-b2-a0
        # Vars: v[a0,b1](+), v[a1,b1](-), v[a1,b2](+), v[a0,b2](-)
        for a0 in range(4):
            for b1 in range(4):
                for a1 in range(4):
                    for b2 in range(4):
                        walks.append([
                            (a0*4+b1, 1),
                            (a1*4+b1, -1),
                            (a1*4+b2, 1),
                            (a0*4+b2, -1)
                        ])
                        
        # Length 6: a0-b1-a1-b2-a2-b3-a0
        # Vars: v[a0,b1](+), v[a1,b1](-), v[a1,b2](+), v[a2,b2](-), v[a2,b3](+), v[a0,b3](-)
        for a0 in range(4):
            for b1 in range(4):
                for a1 in range(4):
                    for b2 in range(4):
                        for a2 in range(4):
                            for b3 in range(4):
                                walks.append([
                                    (a0*4+b1, 1),
                                    (a1*4+b1, -1),
                                    (a1*4+b2, 1),
                                    (a2*4+b2, -1),
                                    (a2*4+b3, 1),
                                    (a0*4+b3, -1)
                                ])
                                
        # Length 8: a0-b1-a1-b2-a2-b3-a3-b4-a0
        # Vars: v[a0,b1](+), v[a1,b1](-), v[a1,b2](+), v[a2,b2](-), v[a2,b3](+), v[a3,b3](-), v[a3,b4](+), v[a0,b4](-)
        for a0 in range(4):
            for b1 in range(4):
                for a1 in range(4):
                    for b2 in range(4):
                        for a2 in range(4):
                            for b3 in range(4):
                                for a3 in range(4):
                                    for b4 in range(4):
                                        walks.append([
                                            (a0*4+b1, 1),
                                            (a1*4+b1, -1),
                                            (a1*4+b2, 1),
                                            (a2*4+b2, -1),
                                            (a2*4+b3, 1),
                                            (a3*4+b3, -1),
                                            (a3*4+b4, 1),
                                            (a0*4+b4, -1)
                                        ])
                                        if len(walks) > 50000: break
                                    if len(walks) > 50000: break
                                if len(walks) > 50000: break
                            if len(walks) > 50000: break
                        if len(walks) > 50000: break
                    if len(walks) > 50000: break
                if len(walks) > 50000: break
            if len(walks) > 50000: break
        return walks

    # To keep memory and time reasonable, we might not need ALL walks if we use a smart search.
    # But 16k walks is fine.
    all_walks = get_closed_walks()
    
    # Map variable index to list of walks it participates in
    var_to_walks = [[] for _ in range(NUM_VARS)]
    for idx, walk in enumerate(all_walks):
        for var_idx, sign in walk:
            var_to_walks[var_idx].append(idx)

    def count_violations(voltages):
        viol_count = 0
        for walk in all_walks:
            s = 0
            for var_idx, sign in walk:
                s += sign * voltages[var_idx]
            if s % MOD == 0:
                viol_count += 1
        return viol_count

    def min_conflicts_search(deadline):
        random.seed(42)
        best_volt = None
        best_viol = float('inf')
        
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            voltages = [random.randint(0, MOD-1) for _ in range(NUM_VARS)]
            viol_count = count_violations(voltages)
            
            if viol_count == 0:
                return voltages
            
            if viol_count < best_viol:
                best_viol = viol_count
                best_volt = voltages[:]
                
            # Local search
            stuck = False
            while not stuck and time.time() < deadline - 10:
                stuck = True
                # Pick a random variable involved in a violation
                # We can just iterate or pick random var
                var_idx = random.randint(0, NUM_VARS - 1)
                
                old_val = voltages[var_idx]
                best_new_val = old_val
                min_viol = viol_count
                
                # Try all values
                for new_val in range(MOD):
                    if new_val == old_val:
                        continue
                    voltages[var_idx] = new_val
                    # Recalculate violations only for walks involving var_idx
                    # For simplicity, just recount all if small, or optimize
                    # Given 20k walks, recounting all is ~0.01s. 
                    # 16 vars * 40 vals * 0.01s = 6.4s per step? Too slow.
                    
                    # Optimization: 
                    # We need a faster violation count.
                    # Let's maintain a boolean array is_violated for each walk.
                    # And a count of violations per variable.
                    
                    pass # Placeholder for optimized logic below
                    
                # Fallback: just use full recount if optimized is too complex to write quickly
                # But we need speed.
                
                # Let's implement the optimized update.
                # This is inside the loop, so we need the state outside.
                
            # End local search
            
        return best_volt

    # Optimized MinConflicts
    def optimized_min_conflicts(deadline):
        random.seed(42)
        # Precompute walk sums structure for fast update?
        # Let's just use the full recount but limit iterations.
        # 20k walks * 16 vars * 40 vals is too much for pure Python.
        
        # Alternative: Use a heuristic.
        # The constraints are linear.
        # We can try to solve a subset of constraints first.
        
        # Let's try a simple random search with a good number of restarts and a quick local search.
        # If local search is too slow, we rely on random restarts hitting a solution.
        # Probability of a random assignment satisfying one constraint is 39/40.
        # For 20k constraints, probability is (39/40)^20000 ~ 0.
        # So random search won't work. We MUST do local search.
        
        # To make local search fast, we must update violations incrementally.
        
        is_violated = [False] * len(all_walks)
        var_viol_count = [0] * NUM_VARS
        
        def update_violations(voltages):
            for i, walk in enumerate(all_walks):
                s = 0
                for var_idx, sign in walk:
                    s += sign * voltages[var_idx]
                if s % MOD == 0:
                    if not is_violated[i]:
                        is_violated[i] = True
                        for var_idx, sign in walk:
                            var_viol_count[var_idx] += 1
                else:
                    if is_violated[i]:
                        is_violated[i] = False
                        for var_idx, sign in walk:
                            var_viol_count[var_idx] -= 1
                            
        def get_total_violations():
            return sum(1 for v in is_violated if v)
            
        best_volt = None
        best_total = float('inf')
        
        for restart in range(100):
            if time.time() > deadline - 5:
                break
                
            voltages = [random.randint(0, MOD-1) for _ in range(NUM_VARS)]
            # Initialize violation states
            for i in range(len(all_walks)):
                is_violated[i] = False
                var_viol_count[i] = 0 # Wait, var_viol_count is per variable, not per walk.
            
            # Reset var_viol_count
            for i in range(NUM_VARS):
                var_viol_count[i] = 0
                
            update_violations(voltages)
            total_viol = get_total_violations()
            
            if total_viol == 0:
                return voltages
                
            if total_viol < best_total:
                best_total = total_viol
                best_volt = voltages[:]
                
            # Local Search
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Find a variable with max violations
                # Or random variable with violations > 0
                candidates = [i for i in range(NUM_VARS) if var_viol_count[i] > 0]
                if not candidates:
                    break
                var_idx = random.choice(candidates)
                
                old_val = voltages[var_idx]
                best_new_val = old_val
                min_local_viol = var_viol_count[var_idx]
                
                for new_val in range(MOD):
                    if new_val == old_val:
                        continue
                    voltages[var_idx] = new_val
                    # Update violations for walks involving var_idx
                    # We need to recalculate sum for those walks
                    # For each walk in var_to_walks[var_idx]:
                    #   Recalculate sum.
                    #   If changed status, update is_violated and var_viol_count.
                    
                    # This is O(degree of var) * O(length of walk)
                    # Degree is ~20000/16 = 1250. Length 4-8.
                    # 1250 * 8 = 10000 ops per value. 40 values = 400k ops.
                    # 16 vars * 400k = 6.4M ops per step.
                    # Python is slow. 6.4M ops might take 1s.
                    # Too slow for many steps.
                    
                    pass # Need a faster way or accept fewer steps.
                
                # If we can't afford the inner loop, we just do a few steps.
                # Let's just do 10 steps of local search per restart.
                break # Simplified for now, will refine if needed.
                
        return best_volt

    # Given the complexity of incremental update in pure Python, 
    # and the small number of variables (16), 
    # we can try a **Genetic Algorithm** or **Simulated Annealing** with full recount.
    # Full recount is 20k walks * 8 ops = 160k ops. ~0.01s.
    # 1000 iterations * 0.01s = 10s.
    # This is feasible!
    
    def simulated_annealing(deadline):
        random.seed(42)
        voltages = [random.randint(0, MOD-1) for _ in range(NUM_VARS)]
        current_viol = count_violations(voltages)
        best_volt = voltages[:]
        best_viol = current_viol
        
        T = 10.0
        T_min = 0.1
        cooling = 0.99
        
        while T > T_min and time.time() < deadline - 5:
            new_volt = voltages[:]
            var_idx = random.randint(0, NUM_VARS - 1)
            new_val = random.randint(0, MOD-1)
            new_volt[var_idx] = new_val
            
            new_viol = count_violations(new_volt)
            
            if new_viol < current_viol or random.random() < pow(2.718, -(new_viol - current_viol) / T):
                voltages = new_volt
                current_viol = new_viol
                
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_volt = voltages[:]
                    
                    if best_viol == 0:
                        return best_volt
                        
            T *= cooling
            
        return best_volt

    candidate_volt = simulated_annealing(deadline)
    
    if candidate_volt is not None:
        voltages = [[candidate_volt[i*4+j] for j in range(4)] for i in range(4)]
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for k in range(40):
                    a_idx = i * 40 + k
                    b_idx = 160 + j * 40 + (k + volt) % 40
                    edges.append((a_idx, b_idx))
                    
        # Verify
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
            
    # Fallback
    return _construct_luw(4, 10, deadline)

def _verify_girth_exact(n, edges, g, deadline):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_len = g - 1
    
    for start_v in range(n):
        if time.time() > deadline:
            return False
            
        states = set()
        for w in adj[start_v]:
            states.add((w, start_v))
            
        for d in range(2, max_len + 1):
            new_states = set()
            for (v, p) in states:
                for w in adj[v]:
                    if w != p:
                        new_states.add((w, v))
            if any(v == start_v for (v, p) in new_states):
                return False
            states = new_states
            if not states:
                break
        
    return True

def _construct_luw(k, g, deadline):
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
