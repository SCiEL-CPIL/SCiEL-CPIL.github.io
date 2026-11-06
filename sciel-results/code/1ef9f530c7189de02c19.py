import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin before 120s limit
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2) configurations
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    def get_pair_partitions(six):
        """Generate all 15 ways to partition 6 vertices into 3 pairs."""
        v = list(six)
        parts = []
        # Fix v[0] paired with v[1..5]
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            rem = [v[j] for j in range(1, 6) if j != i]
            # Pair rem[0] with rem[1], rem[2], or rem[3]
            for j in range(1, 4):
                pair2 = (rem[0], rem[j])
                if j == 1:
                    pair3 = (rem[2], rem[3])
                elif j == 2:
                    pair3 = (rem[1], rem[3])
                else:
                    pair3 = (rem[1], rem[2])
                parts.append([pair1, pair2, pair3])
        return parts
    
    for six_subset in combinations(range(n), 6):
        for partition in get_pair_partitions(six_subset):
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
    
    num_k222s = len(k222_triples)  # 120120
    
    # Count how many K(2,2,2)s each triple is in
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]
    
    class State:
        def __init__(self):
            self.present = [False] * num_triples
            self.k_counts = [0] * num_k222s
            self.size = 0
            self.missing = set(range(num_triples))
        
        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True
        
        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present[t_id] = True
            self.size += 1
            self.missing.discard(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True
        
        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            self.missing.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
        
        def present_list(self):
            return [i for i in range(num_triples) if self.present[i]]
        
        def missing_list(self):
            return list(self.missing)
    
    def polish(state, max_iters=200):
        """Systematically try to add any missing triple that fits."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            missing = state.missing_list()
            for t_id in missing:
                if time.time() > DEADLINE:
                    return
                if state.add(t_id):
                    improved = True
    
    def simulated_annealing(state, steps=5000, t0=1.0, t1=0.01):
        """Simulated annealing with 1-swap and 2-swap moves."""
        import math
        current_size = state.size
        best_size = state.size
        best_present = state.present[:]
        
        for step in range(steps):
            if time.time() > DEADLINE:
                break
            temperature = t0 * (t1 / t0) ** (step / steps)
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.4:
                # 1-swap: remove one, add one
                if not state.missing:
                    continue
                present = state.present_list()
                if not present:
                    continue
                t_out = random.choice(present)
                t_in = random.choice(list(state.missing))
                
                state.remove(t_out)
                if state.add(t_in):
                    delta = 0  # size unchanged
                    accept = True
                else:
                    # Can't add t_in, try to add t_out back
                    state.add(t_out)
                    continue
                
                # Check if we improved
                if state.size > best_size:
                    best_size = state.size
                    best_present = state.present[:]
                
            elif move_type < 0.7:
                # 2-swap: remove two, add two
                if len(state.missing) < 2:
                    continue
                present = state.present_list()
                if len(present) < 2:
                    continue
                t_out1, t_out2 = random.sample(present, 2)
                missing_list = list(state.missing)
                t_in1, t_in2 = random.sample(missing_list, 2)
                
                state.remove(t_out1)
                state.remove(t_out2)
                added1 = state.add(t_in1)
                added2 = state.add(t_in2)
                
                if added1 and added2:
                    if state.size > best_size:
                        best_size = state.size
                        best_present = state.present[:]
                else:
                    # Revert
                    if added1:
                        state.remove(t_in1)
                    if added2:
                        state.remove(t_in2)
                    state.add(t_out2)
                    state.add(t_out1)
                    continue
            else:
                # 3-swap: remove three, add three
                if len(state.missing) < 3:
                    continue
                present = state.present_list()
                if len(present) < 3:
                    continue
                t_outs = random.sample(present, 3)
                missing_list = list(state.missing)
                t_ins = random.sample(missing_list, 3)
                
                for t in t_outs:
                    state.remove(t)
                
                all_added = True
                for t in t_ins:
                    if not state.add(t):
                        all_added = False
                        break
                
                if all_added:
                    if state.size > best_size:
                        best_size = state.size
                        best_present = state.present[:]
                else:
                    # Revert
                    for t in t_ins:
                        if state.present[t]:
                            state.remove(t)
                    for t in t_outs:
                        state.add(t)
                    continue
            
            # Accept with probability based on temperature
            # For this problem, we always accept if size doesn't decrease
            # (since we only add when it fits)
        
        # Restore best if we found one
        if best_size > state.size:
            # Need to rebuild state from best_present
            for i in range(num_triples):
                if best_present[i] and not state.present[i]:
                    state.add(i)
                elif not best_present[i] and state.present[i]:
                    state.remove(i)
        
        return state
    
    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = State()
        order = list(range(num_triples))
        local_rng.shuffle(order)
        # Sort by number of K(2,2,2)s (fewer constraints first)
        order.sort(key=lambda t: triple_k222_count[t])
        # Add some randomness
        for i in range(len(order)):
            if i > 0 and i < len(order):
                swap_idx = i - local_rng.randint(0, min(i, 20))
                order[i], order[swap_idx] = order[swap_idx], order[i]
        for t_id in order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        polish(state)
        return state
    
    def perturb_restart(state, best_present, n_perturb=15):
        """Create a perturbed copy of the best solution."""
        new_state = State()
        for i in range(num_triples):
            if best_present[i]:
                new_state.present[i] = True
                new_state.size += 1
                new_state.missing.discard(i)
        
        # Rebuild k_counts
        for k_idx in range(num_k222s):
            count = 0
            for t_id in k222_triples[k_idx]:
                if new_state.present[t_id]:
                    count += 1
            new_state.k_counts[k_idx] = count
        
        # Perturb: remove some random present triples
        present = new_state.present_list()
        for _ in range(n_perturb):
            if present:
                t = random.choice(present)
                new_state.remove(t)
                present = new_state.present_list()
        
        polish(new_state)
        return new_state
    
    best_size = 0
    best_present = None
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy inits
    for i in range(8):
        if time.time() > DEADLINE:
            break
        state = greedy_init(42 + i * 7919)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
    
    # Phase 2: Simulated annealing from best
    if best_present is not None and time.time() < DEADLINE:
        state = State()
        for i in range(num_triples):
            if best_present[i]:
                state.present[i] = True
                state.size += 1
                state.missing.discard(i)
        for k_idx in range(num_k222s):
            count = 0
            for t_id in k222_triples[k_idx]:
                if state.present[t_id]:
                    count += 1
            state.k_counts[k_idx] = count
        
        # Run SA with multiple restarts
        sa_restarts = 5
        for r in range(sa_restarts):
            if time.time() > DEADLINE:
                break
            # Perturb
            n_perturb = 10 + r * 3
            present = state.present_list()
            for _ in range(n_perturb):
                if present:
                    t = random.choice(present)
                    state.remove(t)
                    present = state.present_list()
            
            polish(state)
            
            # SA
            steps = max(2000, int((DEADLINE - time.time()) * 20))
            state = simulated_annealing(state, steps=steps, t0=1.0, t1=0.01)
            
            if state.size > best_size:
                best_size = state.size
                best_present = state.present[:]
    
    # Phase 3: Final polish
    if best_present is not None:
        state = State()
        for i in range(num_triples):
            if best_present[i]:
                state.present[i] = True
                state.size += 1
                state.missing.discard(i)
        for k_idx in range(num_k222s):
            count = 0
            for t_id in k222_triples[k_idx]:
                if state.present[t_id]:
                    count += 1
            state.k_counts[k_idx] = count
        
        polish(state, max_iters=500)
        
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
    
    return [list(triples[t_id]) for t_id in range(num_triples) if best_present[t_id]]
