import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills after 150s. 
    # We should aim to finish by 145s to be safe.
    DEADLINE = start_time + 145.0

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of points.
    # There are 8 triples in a K(2,2,2).
    # We need to identify all sets of 8 triples that form a K(2,2,2).
    
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    def get_pair_partitions(six):
        """Generate all 15 ways to partition 6 vertices into 3 pairs."""
        v = list(six)
        parts = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            rem = [v[j] for j in range(1, 6) if j != i]
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

    # Build the list of all K(2,2,2) configurations
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

    num_k222s = len(k222_triples)
    # Precompute the number of K(2,2,2)s each triple is part of
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]

    class State:
        def __init__(self):
            self.present = [False] * num_triples
            self.k_counts = [0] * num_k222s
            self.size = 0
            # For fast access to present triples
            self.present_list = []

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
            self.present_list.append(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            # Efficient removal from list: swap with last and pop
            # But order doesn't matter for correctness, only for iteration if we care.
            # For simplicity and safety, we just mark it false. 
            # We will rebuild present_list if needed or just iterate all.
            # Given 560 items, iterating all is fast.
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
            # We don't update present_list immediately to save time, 
            # but we must ensure we don't add duplicates. 
            # can_add checks present[t_id], so it's safe.
            # However, random.choice(present_list) might pick a removed one.
            # So we should clean present_list periodically or use a set.
            # Let's use a set for present_list to allow O(1) removal and safe iteration.
            pass 

        def get_present_list(self):
            return [i for i in range(num_triples) if self.present[i]]

    # Optimization: Use a set for present triples to allow fast random selection
    class OptimizedState:
        def __init__(self):
            self.present = [False] * num_triples
            self.k_counts = [0] * num_k222s
            self.size = 0
            self.present_set = set()

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
            self.present_set.add(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            self.present_set.remove(t_id)
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1

        def get_present_list(self):
            return list(self.present_set)

    def polish(state, max_iters=100000):
        """Add any missing triple that fits. Repeat until no more can be added."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Try to add all missing triples
            # Sort by constraint count to add "easier" ones first? 
            # Or just iterate.
            for t_id in range(num_triples):
                if not state.present[t_id]:
                    if state.add(t_id):
                        improved = True
        return state

    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = OptimizedState()
        # Randomize the order of triples
        order = list(range(num_triples))
        local_rng.shuffle(order)
        for t_id in order:
            if time.time() > DEADLINE - 5:
                break
            state.add(t_id)
        polish(state)
        return state

    def perturb(state, k):
        """Remove k random present triples."""
        present_list = state.get_present_list()
        if len(present_list) <= k:
            return
        to_remove = local_rng.sample(present_list, k)
        for t_id in to_remove:
            state.remove(t_id)

    def iterated_local_search(initial_state, max_time):
        """
        ILS: 
        1. Start with initial_state
        2. Loop:
           a. Perturb current solution
           b. Polish
           c. Accept if better or with prob exp(-delta/T)
        """
        current_state = OptimizedState()
        # Copy initial state
        for i in range(num_triples):
            if initial_state.present[i]:
                current_state.add(i)
        
        best_state = OptimizedState()
        for i in range(num_triples):
            if initial_state.present[i]:
                best_state.add(i)
        
        best_size = best_state.size
        current_size = current_state.size
        
        T = 10.0
        cooling = 0.999
        rng = random.Random(123)
        
        while time.time() < max_time:
            # Perturb
            # Increase perturbation size over time? 
            # Start small, increase if stuck.
            k = rng.randint(2, 8)
            # Occasionally do a larger kick
            if rng.random() < 0.1:
                k = rng.randint(10, 20)
            
            # Create a copy to perturb
            trial_state = OptimizedState()
            for i in range(num_triples):
                if current_state.present[i]:
                    trial_state.add(i)
            
            present_list = trial_state.get_present_list()
            if len(present_list) > k:
                to_remove = rng.sample(present_list, k)
                for t_id in to_remove:
                    trial_state.remove(t_id)
            
            # Polish
            polish(trial_state)
            
            trial_size = trial_state.size
            delta = trial_size - current_size
            
            if delta >= 0:
                # Accept improvement
                current_state = trial_state
                current_size = trial_size
                if current_size > best_size:
                    best_size = current_size
                    # Update best state
                    for i in range(num_triples):
                        best_state.present[i] = current_state.present[i]
                        best_state.k_counts[i] = current_state.k_counts[i] # Wait, k_counts is indexed by k222, not triple
                        # Actually, we just need to copy the present array and rebuild k_counts or copy the whole state
                        # Let's just keep a reference to the best present array
                    best_state_present = current_state.present[:]
                    best_state_size = current_size
            else:
                # Accept worse with probability
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_state = trial_state
                    current_size = trial_size
            
            T *= cooling
            if T < 0.1:
                T = 0.1 # Keep some randomness

        return best_state_present, best_state_size

    # Main Execution
    best_present = None
    best_size = 0
    
    # Phase 1: Multiple Greedy Initializations
    for attempt in range(20):
        if time.time() > DEADLINE - 60:
            break
        state = greedy_init(42 + attempt * 13)
        polish(state)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
    
    if best_present is None:
        # Fallback
        state = greedy_init(1)
        best_present = state.present[:]
        best_size = state.size

    # Phase 2: Iterated Local Search from the best solution
    # We will run ILS for the remaining time
    # We need to pass a state object or create one
    init_state = OptimizedState()
    for i in range(num_triples):
        if best_present[i]:
            init_state.add(i)
            
    # Run ILS
    # We want to return the best found during ILS
    # Modify ILS to return the best state found internally
    
    def run_ils(init_state_obj, time_limit):
        current_state = OptimizedState()
        for i in range(num_triples):
            if init_state_obj.present[i]:
                current_state.add(i)
        
        best_state_present = current_state.present[:]
        best_size = current_state.size
        
        T = 5.0
        cooling = 0.9995
        rng = random.Random(999)
        
        while time.time() < time_limit:
            k = rng.randint(2, 6)
            if rng.random() < 0.05:
                k = rng.randint(8, 15)
            
            trial_state = OptimizedState()
            for i in range(num_triples):
                if current_state.present[i]:
                    trial_state.add(i)
            
            present_list = trial_state.get_present_list()
            if len(present_list) > k:
                to_remove = rng.sample(present_list, k)
                for t_id in to_remove:
                    trial_state.remove(t_id)
            
            polish(trial_state)
            
            trial_size = trial_state.size
            delta = trial_size - current_state.size
            
            if delta >= 0:
                current_state = trial_state
                if current_state.size > best_size:
                    best_size = current_state.size
                    best_state_present = current_state.present[:]
            else:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_state = trial_state
            
            T *= cooling
            if T < 0.05:
                T = 0.05

        return best_state_present, best_size

    # Run ILS with remaining time
    # Leave 5 seconds for final polish and return
    remaining_time = DEADLINE - 5.0
    current_time = time.time()
    ils_deadline = current_time + (remaining_time - current_time) # Just use DEADLINE - 5
    
    best_present, best_size = run_ils(init_state, DEADLINE - 5.0)
    
    # Final Polish
    final_state = OptimizedState()
    for i in range(num_triples):
        if best_present[i]:
            final_state.add(i)
    polish(final_state)
    
    if final_state.size > best_size:
        best_size = final_state.size
        best_present = final_state.present[:]

    return [list(triples[t_id]) for t_id in range(num_triples) if best_present[t_id]]

