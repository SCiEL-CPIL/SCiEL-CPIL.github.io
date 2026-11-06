import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 145s
    DEADLINE = start_time + 145
    
    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is defined by 6 vertices and a pairing.
    # The 8 transversals are the triples formed by picking one vertex from each pair.
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
            # For SA: Track number of "tight" K222s (count == 7)
            self.tight_count = 0
            
        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] >= 7:
                    return False
            return True
        
        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present[t_id] = True
            self.size += 1
            for k_idx in triple_to_k222s[t_id]:
                old_count = self.k_counts[k_idx]
                self.k_counts[k_idx] += 1
                if old_count == 6:
                    self.tight_count += 1
            return True
        
        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            for k_idx in triple_to_k222s[t_id]:
                old_count = self.k_counts[k_idx]
                self.k_counts[k_idx] -= 1
                if old_count == 7:
                    self.tight_count -= 1
        
        def present_list(self):
            return [i for i in range(num_triples) if self.present[i]]
        
        def missing_list(self):
            return [i for i in range(num_triples) if not self.present[i]]
            
    def polish(state, max_iters=1000):
        """Greedy polish: repeatedly try to add any missing triple."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Shuffled order to avoid bias
            missing = state.missing_list()
            random.shuffle(missing)
            for t_id in missing:
                if state.add(t_id):
                    improved = True
                    if time.time() > DEADLINE:
                        return
                    # Restart loop if we added something, as new additions might block others?
                    # No, additions only block. But if we added one, we should check others.
                    # Actually, just continue to next missing in current list.
                    # But the list was generated before. New additions don't make missing ones addable.
                    # So we can break early? No, because order matters?
                    # If A blocks B, and we add A, B is still blocked.
                    # If we didn't add A, B might be addable.
                    # Since we are adding, we are moving towards a local optimum.
                    # We should just do one full pass over missing.
        
    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = State()
        order = list(range(num_triples))
        # Sort by number of K222s (least constrained first)
        order.sort(key=lambda t: triple_k222_count[t])
        # Add some randomness to break ties or explore variations
        for i in range(len(order)):
            if i > 0:
                swap_idx = i - local_rng.randint(0, min(i, 10))
                order[i], order[swap_idx] = order[swap_idx], order[i]
                
        for t_id in order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        polish(state)
        return state

    best_size = 0
    best_present = None
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    kick_interval = 500
    no_improve_counter = 0
    last_best_size = 0
    
    def simulated_annealing(state, seed):
        nonlocal best_size, best_present, no_improve_counter, last_best_size
        local_rng = random.Random(seed)
        T = T_start
        iter_count = 0
        no_improve = 0
        
        while time.time() < DEADLINE:
            iter_count += 1
            
            # 1. Try to Add
            # Pick a random missing triple
            # Optimization: Instead of picking random and checking, we can pick from a pre-computed list?
            # For now, random pick is O(1) check.
            if local_rng.random() < 0.3: # 30% chance to try add
                t_in = local_rng.randint(0, num_triples - 1)
                if not state.present[t_in]:
                    if state.add(t_in):
                        if state.size > best_size:
                            best_size = state.size
                            best_present = state.present[:]
                        no_improve = 0
                        continue
            else:
                # 2. Try to Swap
                # Pick present t_out and missing t_in
                # To make it efficient, pick t_out from present, t_in from missing
                # We need lists. Maintaining lists is O(N) to update.
                # Given N=560, O(N) is fine.
                present_list = state.present_list()
                if not present_list:
                    break
                t_out = local_rng.choice(present_list)
                
                # Find a missing triple
                # To avoid O(N) scan for missing, we can just pick random and check
                # Or maintain a missing list?
                # Let's pick random t_in
                t_in = local_rng.randint(0, num_triples - 1)
                if state.present[t_in]:
                    continue # Skip if present
                
                # Attempt swap
                # Remove t_out
                state.remove(t_out)
                
                # Check if t_in can be added
                if state.can_add(t_in):
                    state.add(t_in)
                    
                    # Calculate Delta Tight Count
                    # We removed t_out and added t_in.
                    # Delta Tight = (New Tight) - (Old Tight)
                    # We didn't store Old Tight explicitly in the swap logic easily.
                    # Let's calculate Delta Tight during the add/remove?
                    # Better: Store tight_count before swap.
                    # Actually, we can just compute the change.
                    # When we removed t_out, tight_count decreased by count of Ks where t_out was in and count was 7.
                    # When we added t_in, tight_count increased by count of Ks where t_in was in and count was 6 (before add).
                    
                    # Let's just use the current state's tight_count vs a hypothetical "old" tight_count?
                    # We didn't save it.
                    # Let's modify add/remove to return delta? Or just calculate it.
                    
                    # Recalculate Delta Tight manually for this specific move
                    delta_tight = 0
                    # Effect of removing t_out:
                    # For each K in t_out, if count[K] (after remove) was 6, it was 7 before. So tight decreased by 1.
                    # So if current count is 6, it was tight before.
                    for k_idx in triple_to_k222s[t_out]:
                        if state.k_counts[k_idx] == 6:
                            delta_tight -= 1 # It was tight, now it isn't. Tight count dropped.
                            
                    # Effect of adding t_in:
                    # For each K in t_in, if count[K] (before add) was 6, it becomes 7.
                    # Current count is after add. So if current count is 7, it was 6 before.
                    for k_idx in triple_to_k222s[t_in]:
                        if state.k_counts[k_idx] == 7:
                            delta_tight += 1 # It wasn't tight, now it is. Tight count rose.
                    
                    # Objective: Minimize Tight Count.
                    # Energy E = Tight Count.
                    # Delta E = delta_tight.
                    # If delta_tight < 0, accept (improvement).
                    # If delta_tight > 0, accept with prob exp(-delta_tight / T).
                    
                    if delta_tight < 0:
                        no_improve = 0
                    else:
                        if delta_tight > 0:
                            prob = math.exp(-delta_tight / T)
                            if local_rng.random() > prob:
                                # Reject swap
                                state.remove(t_in)
                                state.add(t_out)
                                no_improve += 1
                            else:
                                no_improve = 0
                        else: # delta_tight == 0
                            # Accept to explore
                            no_improve = 0
                else:
                    # t_in could not be added, revert
                    state.add(t_out)
                    no_improve += 1
            
            # Cooling
            T *= cooling_rate
            if T < T_end:
                T = T_end
            
            # Kick Restart
            if no_improve > kick_interval:
                # Perturb: Remove 10-20 random present triples
                n_perturb = local_rng.randint(10, 20)
                present_list = state.present_list()
                for _ in range(n_perturb):
                    if present_list:
                        t = local_rng.choice(present_list)
                        state.remove(t)
                        present_list.remove(t)
                # Polish to fill back up
                polish(state)
                no_improve = 0
                T = T_start * 0.5 # Reheat slightly
                
    # Main Loop: Multiple Restarts
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 12345 + restart_count * 101
        
        # Initialize
        state = greedy_init(seed)
        
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
            
        # Run SA
        simulated_annealing(state, seed)
        
        # Final Polish
        polish(state)
        
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
            
        # Check time
        if time.time() > DEADLINE - 5:
            break

    return [list(triples[t_id]) for t_id in range(num_triples) if best_present[t_id]]
