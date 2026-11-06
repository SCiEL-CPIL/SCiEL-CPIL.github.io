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
    
    # Precompute for each triple, which other triples conflict (share a K222 at count 7)
    # This is dynamic, so we'll compute on the fly but cache the k222 lists
    
    class Solver:
        __slots__ = ['present', 'k_counts', 'present_list', 'missing_list']
        
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.present_list = []
            self.missing_list = list(range(num_triples))
            
        def can_add(self, t_id):
            if t_id in self.present:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True
            
        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present.add(t_id)
            self.present_list.append(t_id)
            # Remove from missing
            # We'll rebuild missing periodically
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True
            
        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            self.present_list.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
                
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
    
    # Fast polish: repeatedly try to add missing triples until no more can be added
    def polish(solver):
        while time.time() < DEADLINE:
            # Get current missing list
            missing = solver.get_missing()
            if not missing:
                break
            
            added_any = False
            new_missing = []
            for t_id in missing:
                if time.time() > DEADLINE:
                    break
                if t_id in solver.present:
                    continue
                if solver.add(t_id):
                    added_any = True
                else:
                    new_missing.append(t_id)
            
            if not added_any:
                break
            # Rebuild missing list
            solver.missing_list = new_missing
        
        # Final rebuild
        solver.rebuild_missing()

    # Targeted swap: try to remove a triple and add a better one
    def targeted_swap(solver):
        if not solver.present_list:
            return False
        
        missing = solver.get_missing()
        if not missing:
            return False
        
        # Pick a random triple to remove
        t_out = rng.choice(solver.present_list)
        
        # Try to find a missing triple that can be added after removing t_out
        solver.remove(t_out)
        
        found = False
        for t_in in missing:
            if t_in in solver.present:
                continue
            if solver.add(t_in):
                found = True
                break
        
        if found:
            solver.rebuild_missing()
            return True
        else:
            # Revert
            solver.add(t_out)
            solver.rebuild_missing()
            return False

    # Multi-swap: remove k triples, add k triples
    def multi_swap(solver, k=2):
        if len(solver.present_list) < k:
            return False
        
        missing = solver.get_missing()
        if len(missing) < k:
            return False
        
        # Pick k random triples to remove
        to_remove = rng.sample(solver.present_list, k)
        
        for t in to_remove:
            solver.remove(t)
        
        # Try to add k missing triples
        added = 0
        for t_in in missing:
            if added >= k:
                break
            if t_in in solver.present:
                continue
            if solver.add(t_in):
                added += 1
        
        if added == k:
            solver.rebuild_missing()
            return True
        else:
            # Revert
            for t in to_remove:
                solver.add(t)
            solver.rebuild_missing()
            return False

    def greedy_init(rng, solver):
        order = list(range(num_triples))
        rng.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)

    # Main search loop
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        solver = Solver()
        
        # Greedy Init
        greedy_init(rng, solver)
        
        # Polish
        polish(solver)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
        
        # Local search with various move types
        no_improve_count = 0
        last_best = solver.size()
        
        while time.time() < DEADLINE:
            # Choose a move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Single swap
                if targeted_swap(solver):
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
                    no_improve_count += 1
            elif move_type < 0.7:
                # Multi-swap
                if multi_swap(solver, 2):
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
                    no_improve_count += 1
            else:
                # Random kick: remove several random triples, then polish
                kick_size = rng.randint(3, 6)
                if len(solver.present_list) >= kick_size:
                    to_remove = rng.sample(solver.present_list, kick_size)
                    for t in to_remove:
                        solver.remove(t)
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
                    no_improve_count += 1
            
            # If stuck for too long, restart
            if no_improve_count > 200:
                break
        
        # Check time
        if time.time() > DEADLINE - 5:
            break

    return [list(triples[t_id]) for t_id in best_set]
