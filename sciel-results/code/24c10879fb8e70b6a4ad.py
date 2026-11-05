import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def get_initial_lines():
        # Affine lines in F2^4
        # Points are 0..15
        # A line is {x, y, x^y} where x, y, x^y are distinct
        lines = []
        seen = set()
        for x in range(N):
            for y in range(N):
                if x == y: continue
                z = x ^ y
                if z == x or z == y: continue
                t = tuple(sorted((x, y, z)))
                if t not in seen:
                    seen.add(t)
                    lines.append(t)
        return lines

    def run_local_search(initial_set):
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        
        # Initialize with the provided set
        for t in initial_set:
            if is_valid_addition(triple_to_idx[t]):
                add_triple(triple_to_idx[t])
                current_set.add(triple_to_idx[t])
            else:
                # Should not happen if initial set is valid
                pass
                
        count = len(current_set)
        
        improved = True
        no_improve_count = 0
        max_no_improve = 20
        random.seed(int(time.time() * 1000) % 10000)
                        
        while improved and time.time() < deadline - 5:
            improved = False
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                if count > 10:
                    kick_size = random.randint(1, 3)
                    to_remove = random.sample(list(current_set), kick_size)
                    for t in to_remove:
                        remove_triple(t)
                        current_set.remove(t)
                        count -= 1
                    improved = True
                    no_improve_count = 0
                    continue

            missing = [i for i in range(num_triples) if not present[i]]
            if not missing:
                break
            random.shuffle(missing)
            
            added_any = False
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set.add(t_idx)
                    count += 1
                    added_any = True
                    improved = True
                    no_improve_count = 0
                    if time.time() > deadline - 5:
                        break
            
            if not added_any and time.time() < deadline - 5:
                t_idx = random.choice(missing)
                conflicts = []
                for cid in triple_to_configs[t_idx]:
                    if config_counts[cid] == 7:
                        conflicts.append(cid)
                    
                if conflicts:
                    cid = random.choice(conflicts)
                    present_in_cid = []
                    for t in triples_in_config[cid]:
                        if present[t]:
                            present_in_cid.append(t)
                        
                    if present_in_cid:
                        u_idx = random.choice(present_in_cid)
                        remove_triple(u_idx)
                        current_set.remove(u_idx)
                        count -= 1
                        
                        if is_valid_addition(t_idx):
                            add_triple(t_idx)
                            current_set.add(t_idx)
                            count += 1
                            improved = True
                            no_improve_count = 0
                        else:
                            add_triple(u_idx)
                            current_set.add(u_idx)
                            count += 1
                            
        return count, list(current_set)

    # Get the 35 affine lines
    initial_lines = get_initial_lines()
    
    best_count = 0
    best_set = []
    
    # Run local search starting from the algebraic structure
    # We can also try random restarts if time permits
    # But starting from 35 is much better than starting from 0
    
    count, s = run_local_search(initial_lines)
    if count > best_count:
        best_count = count
        best_set = s
        
    # If time remains, try a few random restarts to see if we can beat the algebraic start
    # The algebraic start is likely optimal or near-optimal, so random restarts might not help
    # but they are cheap to try if fast
    
    num_restarts = 0
    while time.time() < deadline - 10:
        seed = num_restarts
        # Random greedy start
        present[:] = False
        config_counts[:] = 0
        current_set = set()
        order = list(range(num_triples))
        random.shuffle(order)
        for t_idx in order:
            if is_valid_addition(t_idx):
                add_triple(t_idx)
                current_set.add(t_idx)
        
        # Local search from random start
        # Reuse the logic? The function above takes initial_set of triples.
        # Let's convert current_set to triples
        init_triples = [all_triples[i] for i in current_set]
        count2, s2 = run_local_search(init_triples)
        
        if count2 > best_count:
            best_count = count2
            best_set = s2
            
        num_restarts += 1
        if num_restarts > 5: # Limit restarts to avoid overhead
            break

    result = [all_triples[i] for i in best_set]
    
    return result
