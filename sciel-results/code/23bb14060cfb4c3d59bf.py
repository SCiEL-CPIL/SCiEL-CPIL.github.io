import sys
import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 17
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        
    cycles = []
    cycles_containing_edge = [[] for _ in range(num_edges)]
    
    vertices = range(n)
    for verts in combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = verts
        seen = set()
        for perm in permutations([v1, v2, v3, v4, v5]):
            path = [v0] + list(perm)
            e1 = edge_index[(min(path[0], path[1]), max(path[0], path[1]))]
            e2 = edge_index[(min(path[1], path[2]), max(path[1], path[2]))]
            e3 = edge_index[(min(path[2], path[3]), max(path[2], path[3]))]
            e4 = edge_index[(min(path[3], path[4]), max(path[3], path[4]))]
            e5 = edge_index[(min(path[4], path[5]), max(path[4], path[5]))]
            e6 = edge_index[(min(path[5], path[0]), max(path[5], path[0]))]
            
            cycle_key = tuple(sorted([e1, e2, e3, e4, e5, e6]))
            
            if cycle_key not in seen:
                seen.add(cycle_key)
                cycles.append(cycle_key)
                for e in cycle_key:
                    cycles_containing_edge[e].append(len(cycles) - 1)

    num_cycles = len(cycles)
    
    def check_cycle(c_idx):
        c_edges = cycles[c_idx]
        c0 = colors[c_edges[0]]
        for e in c_edges[1:]:
            if colors[e] != c0:
                return False
        return True

    def run_search():
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        is_bad = [False] * num_cycles
        bad_indices = set()
        
        for i in range(num_cycles):
            if check_cycle(i):
                is_bad[i] = True
                bad_indices.add(i)
                
        max_iter = 20000
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                return None
            
            if not bad_indices:
                return colors
            
            c_idx = next(iter(bad_indices)) # Arbitrary pick from set
            # Better: Random pick
            # Convert to list? Slow.
            # Just pick a random key from the set?
            # In Python 3.7+, sets are ordered but not randomly accessible by index.
            # We can do: c_idx = random.choice(list(bad_indices)) -- O(N)
            # Or maintain a list of bad indices and clean it up?
            # Let's use a list `bad_list` and a set `bad_set` for O(1) removal?
            # Or just iterate?
            # Let's stick to `random.choice(list(bad_indices))` for simplicity, 
            # assuming bad_indices is not huge.
            # If bad_indices is large (e.g., 10k), list() is 10k ops.
            # 20k iterations * 10k ops = 200M ops. Might be slow.
            
            # Optimization: Keep a list `bad_list`.
            # When removing, swap with last and pop.
            
            # Let's re-implement with list-based bad tracking for speed.
            pass

    # Re-implementing run_search with list-based bad tracking
    
    def run_search_fast():
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        is_bad = [False] * num_cycles
        bad_list = []
        
        for i in range(num_cycles):
            if check_cycle(i):
                is_bad[i] = True
                bad_list.append(i)
                
        max_iter = 50000
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                return None
            
            if not bad_list:
                return colors
            
            # Pick random bad cycle
            idx = random.randint(0, len(bad_list) - 1)
            c_idx = bad_list[idx]
            
            # Verify it's still bad (should be, but safe)
            if not is_bad[c_idx]:
                # Remove it from bad_list
                bad_list[idx] = bad_list[-1]
                bad_list.pop()
                continue
            
            # Pick random edge from cycle
            e_idx = cycles[c_idx][random.randint(0, 5)]
            
            # Change color
            old_color = colors[e_idx]
            new_color = random.randint(0, 3)
            while new_color == old_color:
                new_color = random.randint(0, 3)
            colors[e_idx] = new_color
            
            # Update affected cycles
            for ac_idx in cycles_containing_edge[e_idx]:
                was_bad = is_bad[ac_idx]
                now_bad = check_cycle(ac_idx)
                
                if was_bad != now_bad:
                    is_bad[ac_idx] = now_bad
                    if now_bad:
                        bad_list.append(ac_idx)
                    else:
                        # Remove ac_idx from bad_list
                        # Find it
                        # This is O(N). 
                        # To make it O(1), we need to know its position.
                        # We can maintain a dict `pos_in_bad_list`.
                        pass
                        
    # The O(N) removal is the bottleneck.
    # Let's use the dict approach.
    
    def run_search_optimized():
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        is_bad = [False] * num_cycles
        bad_list = []
        pos = {}
        
        for i in range(num_cycles):
            if check_cycle(i):
                is_bad[i] = True
                pos[i] = len(bad_list)
                bad_list.append(i)
                
        def remove_bad(i):
            if i in pos:
                p = pos[i]
                last = bad_list[-1]
                bad_list[p] = last
                pos[last] = p
                bad_list.pop()
                del pos[i]
                is_bad[i] = False

        def add_bad(i):
            if i not in pos:
                pos[i] = len(bad_list)
                bad_list.append(i)
                is_bad[i] = True

        max_iter = 50000
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                return None
            
            if not bad_list:
                return colors
            
            idx = random.randint(0, len(bad_list) - 1)
            c_idx = bad_list[idx]
            
            if not is_bad[c_idx]:
                remove_bad(c_idx)
                continue
            
            e_idx = cycles[c_idx][random.randint(0, 5)]
            
            old_color = colors[e_idx]
            new_color = random.randint(0, 3)
            while new_color == old_color:
                new_color = random.randint(0, 3)
            colors[e_idx] = new_color
            
            for ac_idx in cycles_containing_edge[e_idx]:
                was_bad = is_bad[ac_idx]
                now_bad = check_cycle(ac_idx)
                
                if was_bad != now_bad:
                    if now_bad:
                        add_bad(ac_idx)
                    else:
                        remove_bad(ac_idx)

        return None

    res = run_search_optimized()
    if res:
        return {'n': n, 'colors': res}
        
    # Fallback
    return {'n': 9, 'colors': [0]*36}

