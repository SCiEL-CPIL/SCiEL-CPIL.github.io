import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118  # 32s safety margin before 150s kill
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2) configurations
    # Each K(2,2,2) is defined by 3 disjoint pairs, giving 8 transversal triples
    k222_triples = []  # list of lists of triple IDs
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_partitions(vertices):
        """Get all ways to partition 6 vertices into 3 pairs."""
        v = list(vertices)
        partitions = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            remaining = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (remaining[0], remaining[j])
                remaining2 = [remaining[k] for k in range(1, 4) if k != j]
                pair3 = (remaining2[0], remaining2[1])
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
    
    # Solver class with efficient constraint tracking
    class Solver:
        def __init__(self):
            self.present = set()
            self.k_counts = [0] * num_k222s
        
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
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True
        
        def remove(self, t_id):
            if t_id not in self.present:
                return
            self.present.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
        
        def size(self):
            return len(self.present)
    
    best_size = 0
    best_set = set()
    
    def polish(solver, max_iterations=50):
        """Add any missing triple that fits, repeating until no improvement."""
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < DEADLINE:
            improved = False
            iterations += 1
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    if solver.add(t_id):
                        improved = True
            if time.time() > DEADLINE:
                return
    
    def random_swap(solver, rng):
        """Try a 1-swap: remove one triple, add another."""
        if not solver.present:
            return False
        
        present_list = list(solver.present)
        
        # Prefer removing triples in tight K(2,2,2)s
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
        
        # Find candidates that can now be added
        candidates = []
        for t_id in range(num_triples):
            if t_id not in solver.present and t_id != t_out:
                if solver.can_add(t_id):
                    candidates.append(t_id)
        
        if not candidates:
            solver.add(t_out)
            return False
        
        t_in = rng.choice(candidates)
        if solver.add(t_in):
            return True
        else:
            solver.add(t_out)
            return False
    
    def random_2swap(solver, rng):
        """Try a 2-swap: remove two triples, add two others."""
        if len(solver.present) < 2:
            return False
        
        present_list = list(solver.present)
        
        # Prefer removing pairs from tight K(2,2,2)s
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
        
        # Find candidates that can now be added
        candidates = []
        for t_id in range(num_triples):
            if t_id not in solver.present and t_id != t_out1 and t_id != t_out2:
                if solver.can_add(t_id):
                    candidates.append(t_id)
        
        if len(candidates) < 2:
            solver.add(t_out1)
            solver.add(t_out2)
            return False
        
        t_in1, t_in2 = rng.sample(candidates, 2)
        ok1 = solver.add(t_in1)
        ok2 = solver.add(t_in2)
        
        if ok1 and ok2:
            return True
        else:
            solver.remove(t_in1)
            solver.remove(t_in2)
            solver.add(t_out1)
            solver.add(t_out2)
            return False
    
    def aggressive_improve(solver, rng, max_attempts=100):
        """Try removing one triple and adding multiple."""
        if not solver.present:
            return False
        
        improved = False
        for _ in range(max_attempts):
            if time.time() > DEADLINE:
                break
            
            present_list = list(solver.present)
            if not present_list:
                break
            
            t_out = rng.choice(present_list)
            solver.remove(t_out)
            
            # Find all triples that can now be added
            addable = []
            for t_id in range(num_triples):
                if t_id not in solver.present:
                    if solver.can_add(t_id):
                        addable.append(t_id)
            
            if len(addable) >= 2:
                # Try adding two
                t_in1, t_in2 = rng.sample(addable, 2)
                ok1 = solver.add(t_in1)
                ok2 = solver.add(t_in2)
                
                if ok1 and ok2:
                    improved = True
                else:
                    solver.remove(t_in1)
                    solver.remove(t_in2)
                    solver.add(t_out)
            elif len(addable) == 1:
                t_in1 = addable[0]
                if solver.add(t_in1):
                    improved = True
                else:
                    solver.add(t_out)
            else:
                solver.add(t_out)
        
        return improved
    
    # Main search loop
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    solver = Solver()
    
    # Start with random ordering
    order = list(range(num_triples))
    rng.shuffle(order)
    
    for t_id in order:
        if time.time() > DEADLINE:
            break
        solver.add(t_id)
    
    polish(solver)
    
    if solver.size() > best_size:
        best_size = solver.size()
        best_set = set(solver.present)
    
    # Phase 2: Local search with swaps
    step = 0
    while time.time() < DEADLINE:
        step += 1
        
        # Choose operation
        op = rng.random()
        if op < 0.4:
            random_swap(solver, rng)
        elif op < 0.7:
            random_2swap(solver, rng)
        elif op < 0.9:
            aggressive_improve(solver, rng, max_attempts=50)
        else:
            polish(solver)
        
        # Update best
        if solver.size() > best_size:
            best_size = solver.size()
            best_set = set(solver.present)
        
        # Periodic restart if stuck
        if step % 500 == 0:
            # Try a fresh start with different seed
            fresh_solver = Solver()
            fresh_order = list(range(num_triples))
            rng.shuffle(fresh_order)
            for t_id in fresh_order:
                if time.time() > DEADLINE:
                    break
                fresh_solver.add(t_id)
            polish(fresh_solver)
            
            if fresh_solver.size() > best_size:
                best_size = fresh_solver.size()
                best_set = set(fresh_solver.present)
                solver = fresh_solver
            elif fresh_solver.size() > solver.size() - 5:
                # Accept if close, to explore different region
                if rng.random() < 0.3:
                    solver = fresh_solver
    
    # Final polish on best solution
    final_solver = Solver()
    final_solver.present = set(best_set)
    for k_idx in range(num_k222s):
        count = 0
        for t_id in k222_triples[k_idx]:
            if t_id in final_solver.present:
                count += 1
        final_solver.k_counts[k_idx] = count
    
    polish(final_solver, max_iterations=100)
    
    if final_solver.size() > best_size:
        best_size = final_solver.size()
        best_set = set(final_solver.present)
    
    # Return result
    result = [list(triples[t_id]) for t_id in sorted(best_set)]
    return result
