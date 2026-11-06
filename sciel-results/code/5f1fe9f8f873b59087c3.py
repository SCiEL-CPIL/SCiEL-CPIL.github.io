import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
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
    
    k222_to_triples = [set(t_ids) for t_ids in k222_triples]
    
    triple_neighbors = [set() for _ in range(num_triples)]
    for k_idx, t_ids in enumerate(k222_triples):
        t_list = list(t_ids)
        for i in range(len(t_list)):
            for j in range(i+1, 8):
                triple_neighbors[t_list[i]].add(t_list[j])
                triple_neighbors[t_list[j]].add(t_list[i])
    
    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
            self.block_count = [0] * num_triples
            self.unblocked_missing = set()
            
        def init_from_set(self, present_set):
            self.present = set(present_set)
            self.k_counts = [0] * num_k222s
            for t_id in self.present:
                for k_idx in triple_to_k222s[t_id]:
                    self.k_counts[k_idx] += 1
            self.block_count = [0] * num_triples
            self.unblocked_missing = set()
            for t_id in range(num_triples):
                if t_id in self.present:
                    continue
                bc = 0
                for k_idx in triple_to_k222s[t_id]:
                    if self.k_counts[k_idx] == 7:
                        bc += 1
                self.block_count[t_id] = bc
                if bc == 0:
                    self.unblocked_missing.add(t_id)
            
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
            self.unblocked_missing.discard(t_id)
            for k_idx in triple_to_k222s[t_id]:
                old_count = self.k_counts[k_idx]
                self.k_counts[k_idx] = old_count + 1
                if old_count == 6:
                    for other_t in k222_to_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.block_count[other_t] += 1
                            if self.block_count[other_t] == 1:
                                self.unblocked_missing.discard(other_t)
            return True
            
        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.discard(t_id)
            for k_idx in triple_to_k222s[t_id]:
                old_count = self.k_counts[k_idx]
                self.k_counts[k_idx] = old_count - 1
                if old_count == 7:
                    for other_t in k222_to_triples[k_idx]:
                        if other_t != t_id and other_t not in self.present:
                            self.block_count[other_t] -= 1
                            if self.block_count[other_t] == 0:
                                self.unblocked_missing.add(other_t)
            bc = 0
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    bc += 1
            self.block_count[t_id] = bc
            if bc == 0:
                self.unblocked_missing.add(t_id)
            else:
                self.unblocked_missing.discard(t_id)
                
        def size(self):
            return len(self.present)

    best_size = 0
    best_set = set()
    
    def polish(solver, max_iterations=20):
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < DEADLINE:
            improved = False
            iterations += 1
            candidates = list(solver.unblocked_missing)
            random.Random(42).shuffle(candidates)
            for t_id in candidates:
                if t_id in solver.present:
                    continue
                if solver.can_add(t_id):
                    solver.add(t_id)
                    improved = True
        return improved

    def random_swap(solver, rng):
        if not solver.present:
            return False
        
        present_list = list(solver.present)
        
        tight_triples = []
        for t_id in present_list:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight_triples.append(t_id)
                    break
        
        if tight_triples and rng.random() < 0.7:
            t_out = rng.choice(tight_triples)
        else:
            t_out = rng.choice(present_list)
        
        solver.remove(t_out)
        
        candidates = []
        for nb in triple_neighbors[t_out]:
            if nb not in solver.present:
                if solver.can_add(nb):
                    candidates.append(nb)
        
        if len(candidates) < 5:
            non_present = [t for t in range(num_triples) if t not in solver.present and t not in set(candidates)]
            sample_size = min(20, len(non_present))
            if sample_size > 0:
                sample = rng.sample(non_present, sample_size)
                for t_id in sample:
                    if solver.can_add(t_id):
                        candidates.append(t_id)
        
        if not candidates:
            solver.add(t_out)
            return False
        
        if len(candidates) > 1:
            scored = [(len(triple_to_k222s[t_id]), t_id) for t_id in candidates]
            scored.sort(reverse=True)
            top_k = min(5, len(scored))
            t_in = scored[rng.randint(0, top_k-1)][1]
        else:
            t_in = candidates[0]
        
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False

    def random_2swap(solver, rng):
        if len(solver.present) < 2:
            return False
        
        present_list = list(solver.present)
        
        tight_triples = []
        for t_id in present_list:
            for k_idx in triple_to_k222s[t_id]:
                if solver.k_counts[k_idx] == 7:
                    tight_triples.append(t_id)
                    break
        
        if len(tight_triples) >= 2 and rng.random() < 0.6:
            t_out1, t_out2 = rng.sample(tight_triples, 2)
        elif len(tight_triples) >= 1 and rng.random() < 0.8:
            t_out1 = rng.choice(tight_triples)
            t_out2 = rng.choice([t for t in present_list if t != t_out1])
        else:
            t_out1, t_out2 = rng.sample(present_list, 2)
        
        solver.remove(t_out1)
        solver.remove(t_out2)
        
        candidates = []
        neighbor_set = set()
        for nb in triple_neighbors[t_out1]:
            if nb not in solver.present and nb != t_out2 and nb not in neighbor_set:
                neighbor_set.add(nb)
                if solver.can_add(nb):
                    candidates.append(nb)
        for nb in triple_neighbors[t_out2]:
            if nb not in solver.present and nb != t_out1 and nb not in neighbor_set:
                neighbor_set.add(nb)
                if solver.can_add(nb):
                    candidates.append(nb)
        
        if len(candidates) < 4:
            non_present = [t for t in range(num_triples) if t not in solver.present and t not in neighbor_set]
            sample_size = min(15, len(non_present))
            if sample_size > 0:
                sample = rng.sample(non_present, sample_size)
                for t_id in sample:
                    if solver.can_add(t_id):
                        candidates.append(t_id)
        
        if len(candidates) < 2:
            solver.add(t_out1)
            solver.add(t_out2)
            return False
        
        scored = [(len(triple_to_k222s[t_id]), t_id) for t_id in candidates]
        scored.sort(reverse=True)
        
        top_k = min(10, len(scored))
        t_in1 = scored[rng.randint(0, top_k-1)][1]
        
        if solver.add(t_in1):
            remaining = [t_id for _, t_id in scored if t_id != t_in1]
            if remaining:
                t_in2 = remaining[rng.randint(0, min(5, len(remaining))-1)]
                if solver.add(t_in2):
                    return True
                else:
                    solver.remove(t_in1)
                    solver.add(t_out1)
                    solver.add(t_out2)
                    return False
            else:
                solver.remove(t_in1)
                solver.add(t_out1)
                solver.add(t_out2)
                return False
        else:
            solver.add(t_out1)
            solver.add(t_out2)
            return False

    def aggressive_improve(solver, rng, max_attempts=50):
        if not solver.present:
            return False
        
        improved = False
        for _ in range(max_attempts):
            if time.time() > DEADLINE:
                break
            
            present_list = list(solver.present)
            if not present_list:
                break
            
            tight_triples = []
            for t_id in present_list:
                for k_idx in triple_to_k222s[t_id]:
                    if solver.k_counts[k_idx] == 7:
                        tight_triples.append(t_id)
                        break
            
            if tight_triples and rng.random() < 0.7:
                t_out = rng.choice(tight_triples)
            else:
                t_out = rng.choice(present_list)
            
            solver.remove(t_out)
            
            addable = []
            for nb in triple_neighbors[t_out]:
                if nb not in solver.present:
                    if solver.can_add(nb):
                        addable.append(nb)
            
            if len(addable) < 3:
                non_present = [t for t in range(num_triples) if t not in solver.present and t not in set(addable)]
                sample_size = min(10, len(non_present))
                if sample_size > 0:
                    sample = rng.sample(non_present, sample_size)
                    for t_id in sample:
                        if solver.can_add(t_id):
                            addable.append(t_id)
            
            if len(addable) >= 2:
                scored = [(len(triple_to_k222s[t_id]), t_id) for t_id in addable]
                scored.sort(reverse=True)
                
                found = False
                for i in range(min(15, len(scored))):
                    t_in1 = scored[i][1]
                    ok1 = solver.add(t_in1)
                    if not ok1:
                        continue
                    
                    for j in range(min(15, len(scored))):
                        if j == i:
                            continue
                        t_in2 = scored[j][1]
                        if solver.add(t_in2):
                            found = True
                            break
                        else:
                            solver.remove(t_in2)
                    
                    if found:
                        improved = True
                        break
                    else:
                        solver.remove(t_in1)
                
                if not found:
                    solver.add(t_out)
            else:
                solver.add(t_out)
        
        return improved

    num_restarts = 30
    time_per_restart = 100.0 / num_restarts
    
    for seed in range(num_restarts):
        if time.time() > DEADLINE:
            break
        
        rng = random.Random(seed * 7919 + 1)
        solver = Solver()
        
        order = list(range(num_triples))
        rng.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE:
                break
            solver.add(t_id)
            
        solver.init_from_set(solver.present)
        
        polish(solver)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
        
        restart_deadline = start_time + (seed + 1) * time_per_restart + 2
        
        while time.time() < restart_deadline and time.time() < DEADLINE:
            for _ in range(25):
                if time.time() > restart_deadline or time.time() > DEADLINE:
                    break
                if rng.random() < 0.5:
                    random_swap(solver, rng)
                else:
                    random_2swap(solver, rng)
            
            polish(solver, max_iterations=5)
            
            if rng.random() < 0.3:
                aggressive_improve(solver, rng, max_attempts=30)
            
            if solver.size() > best_size:
                best_size = solver.size()
                best_set = set(solver.present)
        
        polish(solver, max_iterations=15)
        
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
    
    if best_set:
        final_solver = Solver()
        final_solver.init_from_set(best_set)
        
        polish(final_solver, max_iterations=200)
        
        final_rng = random.Random(12345)
        final_deadline = DEADLINE - 3
        while time.time() < final_deadline:
            if final_rng.random() < 0.5:
                random_swap(final_solver, final_rng)
            else:
                random_2swap(final_solver, final_rng)
            polish(final_solver, max_iterations=10)
            
            if final_solver.size() > best_size:
                best_size = final_solver.size()
                best_set = set(final_solver.present)
        
        polish(final_solver, max_iterations=100)
        
        if final_solver.size() > best_size:
            best_size = final_solver.size()
            best_set = set(final_solver.present)
    
    return [list(triples[t_id]) for t_id in best_set]
