import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2)s
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_partitions(vertices):
        v = list(vertices)
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                if j == 1:
                    pair3 = (remaining[2], remaining[3])
                elif j == 2:
                    pair3 = (remaining[1], remaining[3])
                else:
                    pair3 = (remaining[1], remaining[2])
                partitions.append([pair1, pair2, pair3])
        return partitions

    for six_subset in combinations(range(n), 6):
        for partition in get_partitions(six_subset):
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)
            
            k_idx = len(k222_triples)
            k222_triples.append(t_ids)
            
            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)
                
    num_k222s = len(k222_triples)
    triple_k_degree = [len(k_list) for k_list in triple_to_k222s]
    
    class Solver:
        __slots__ = ['present', 'k_counts', 'present_list', 'missing_list', 'blocked']
        
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.present_list = []
            self.missing_list = list(range(num_triples))
            self.blocked = [False] * num_triples
            
        def can_add(self, t_id):
            if t_id in self.present:
                return True
            if self.blocked[t_id]:
                return False
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True
            
        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present.add(t_id)
            self.present_list.append(t_id)
            self.blocked[t_id] = False
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
                if self.k_counts[k_idx] == 7:
                    for other_t in k222_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.blocked[other_t] = True
            return True
            
        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            self.present_list.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
                if self.k_counts[k_idx] == 6:
                    for other_t in k222_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.blocked[other_t] = False
            self.blocked[t_id] = False
            
        def size(self):
            return len(self.present)
        
        def rebuild_missing(self):
            self.missing_list = [t for t in range(num_triples) if t not in self.present]
            
        def get_missing(self):
            if not self.missing_list:
                self.rebuild_missing()
            return self.missing_list

    best_size = 0
    best_set = set()
    
    rng = random.Random(42)
    
    def polish(solver, max_rounds=20):
        for _ in range(max_rounds):
            if time.time() > DEADLINE:
                break
            added_any = False
            solver.missing_list.sort(key=lambda x: triple_k_degree[x])
            
            new_missing = []
            for t_id in solver.missing_list:
                if time.time() > DEADLINE:
                    break
                if t_id in solver.present:
                    continue
                if not solver.blocked[t_id]:
                    can = True
                    for k_idx in triple_to_k222s[t_id]:
                        if solver.k_counts[k_idx] == 7:
                            can = False
                            break
                    if can:
                        solver.add(t_id)
                        added_any = True
                    else:
                        new_missing.append(t_id)
                else:
                    new_missing.append(t_id)
            
            if not added_any:
                break
            solver.missing_list = new_missing
        
        solver.rebuild_missing()
        for t_id in solver.missing_list:
            if not solver.blocked[t_id]:
                 continue
            is_blocked = False
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    is_blocked = True
                    break
            if not is_blocked:
                solver.blocked[t_id] = False

    def greedy_init(rng, solver):
        order = sorted(range(num_triples), key=lambda x: (triple_k_degree[x], rng.random()))
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)

    def try_swap(solver, t_out, t_in):
        if t_out not in solver.present or t_in in solver.present:
            return False
        solver.remove(t_out)
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False

    def targeted_swap(solver, t_out, rng):
        # Find missing triples that share a K with t_out
        candidates = []
        for k_idx in triple_to_k222s[t_out]:
            for t_in in k222_triples[k_idx]:
                if t_in not in solver.present and not solver.blocked[t_in]:
                    candidates.append(t_in)
        
        if not candidates:
            # Fallback to random missing
            if solver.missing_list:
                t_in = rng.choice(solver.missing_list)
                return try_swap(solver, t_out, t_in)
            return False
        
        # Pick a random candidate
        t_in = rng.choice(candidates)
        return try_swap(solver, t_out, t_in)

    def multi_swap(solver, k_remove, rng):
        if len(solver.present_list) < k_remove:
            return False
        to_remove = rng.sample(solver.present_list, k_remove)
        for t in to_remove:
            solver.remove(t)
        
        added = 0
        missing_sorted = sorted(solver.missing_list, key=lambda x: triple_k_degree[x])
        for t_in in missing_sorted:
            if added >= k_remove:
                break
            if t_in in solver.present:
                continue
            if solver.add(t_in):
                added += 1
        
        if added == k_remove:
            solver.rebuild_missing()
            return True
        else:
            for t in to_remove:
                solver.add(t)
            solver.rebuild_missing()
            return False

    def tightness_kick(solver, kick_size, rng):
        if len(solver.present_list) < kick_size:
            return False
        
        # Calculate tightness
        tightness = [0] * num_triples
        for t_id in solver.present_list:
            count = 0
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] >= 6:
                    count += 1
            tightness[t_id] = count
        
        # Select top kick_size by tightness, with random tie-breaking
        present_with_tightness = [(tightness[t], t) for t in solver.present_list]
        present_with_tightness.sort(key=lambda x: (-x[0], rng.random()))
        to_remove = [t for _, t in present_with_tightness[:kick_size]]
        
        for t in to_remove:
            solver.remove(t)
        
        polish(solver)
        return True

    def simulated_annealing_step(solver, temp, rng):
        move_type = rng.random()
        
        if move_type < 0.6:
            # Targeted Swap
            if not solver.present_list:
                return False
            t_out = rng.choice(solver.present_list)
            return targeted_swap(solver, t_out, rng)
        elif move_type < 0.8:
            # Multi-swap with 2
            return multi_swap(solver, 2, rng)
        elif move_type < 0.9:
            # Multi-swap with 3
            return multi_swap(solver, 3, rng)
        else:
            # Tightness Kick
            kick_size = rng.randint(4, 8)
            return tightness_kick(solver, kick_size, rng)

    def kick(solver, kick_size, rng):
        if len(solver.present_list) < kick_size:
            return False
        to_remove = rng.sample(solver.present_list, kick_size)
        for t in to_remove:
            solver.remove(t)
        polish(solver)
        return True

    # Main search loop
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        solver = Solver()
        
        greedy_init(rng, solver)
        polish(solver)
        
        current_best = solver.size()
        if current_best > best_size:
            best_size = current_best
            best_set = set(solver.present)
        
        temp = 1.0
        cooling = 0.99
        no_improve_count = 0
        last_best = solver.size()
        
        while time.time() < DEADLINE:
            if simulated_annealing_step(solver, temp, rng):
                polish(solver)
                
                if solver.size() > best_size:
                    best_size = solver.size()
                    best_set = set(solver.present)
                
                if solver.size() > last_best:
                    last_best = solver.size()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                if rng.random() < temp:
                    polish(solver)
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            
            temp *= cooling
            
            if no_improve_count > 50:
                kick_size = rng.randint(4, 8)
                if kick(solver, kick_size, rng):
                    no_improve_count = 0
                    temp = max(temp, 0.5)
            
            if no_improve_count > 200:
                break
        
        if time.time() > DEADLINE - 5:
            break

    return [list(triples[t_id]) for t_id in best_set]
