"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_value(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _optimize_edges(values, m=35, deadline=None):
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    best_raw = float('inf')
    best_edges = None
    
    # Multiple random restarts
    num_restarts = 50
    for r in range(num_restarts):
        if deadline and time.time() > deadline - 2:
            break
        
        random.shuffle(all_edges)
        current = list(all_edges[:m])
        current_raw = _raw_value(values, current)
        
        if current_raw < best_raw:
            best_raw = current_raw
            best_edges = list(current)
        
        # Local search: try swaps
        improved = True
        while improved and (deadline is None or time.time() < deadline - 2):
            improved = False
            # Compute current sums and prods
            cur_sums = {}
            cur_prods = {}
            for i, j in current:
                s = values[i] + values[j]
                p = values[i] * values[j]
                cur_sums[s] = cur_sums.get(s, 0) + 1
                cur_prods[p] = cur_prods.get(p, 0) + 1
            
            # Try each edge in current against each edge not in current
            current_set = set(tuple(e) for e in current)
            non_current = [e for e in all_edges if tuple(e) not in current_set]
            random.shuffle(non_current)
            
            for idx, e_in in enumerate(current):
                if deadline and time.time() > deadline - 2:
                    break
                # Try swapping e_in with a few non-current edges
                for e_out in non_current[:20]:
                    new_edges = [current[k] if k != idx else e_out for k in range(m)]
                    new_raw = _raw_value(values, new_edges)
                    if new_raw < current_raw:
                        current = new_edges
                        current_raw = new_raw
                        improved = True
                        if current_raw < best_raw:
                            best_raw = current_raw
                            best_edges = list(current)
                        break  # restart non_current loop
    
    return best_edges, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Candidate value sets
    value_sets = [
        list(range(1, 15)),           # 1-14
        list(range(2, 16)),           # 2-15
        [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
        [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192],  # powers of 2
        [1,2,3,5,7,11,13,17,19,23,29,31,37,41],  # primes
        [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        [1,2,3,4,6,8,9,12,16,18,24,36,48,72],  # divisors
    ]
    
    best_overall = float('inf')
    best_result = None
    
    for vs in value_sets:
        if time.time() > deadline - 5:
            break
        edges, raw = _optimize_edges(vs, 35, deadline)
        if raw < best_overall:
            best_overall = raw
            best_result = {'values': list(vs), 'edges': [list(e) for e in edges]}
    
    if best_result is None:
        # Fallback
        vals = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_result = {'values': vals, 'edges': edges}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END

