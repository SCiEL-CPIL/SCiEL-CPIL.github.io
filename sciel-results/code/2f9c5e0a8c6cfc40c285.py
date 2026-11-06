import time
import random
from itertools import combinations
import numpy as np
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118  # Safety margin before 150s limit
    
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
    
    # Count how many K(2,2,2)s each triple is in
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]
    
    # Convert to numpy for faster operations
    k222_triples_np = [np.array(t_ids) for t_ids in k222_triples]
    triple_to_k222s_np = [np.array(lst) for lst in triple_to_k222s]
    
    class State:
        def __init__(self):
            self.present = np.zeros(num_triples, dtype=bool)
            self.k_counts = np.zeros(num_k222s, dtype=np.int32)
            self.size = 0
        
        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            ks = triple_to_k222s_np[t_id]
            if len(ks) == 0:
                return True
            return np.all(self.k_counts[ks] < 7)
        
        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present[t_id] = True
            self.size += 1
            ks = triple_to_k222s_np[t_id]
            if len(ks) > 0:
                self.k_counts[ks] += 1
            return True
        
        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            ks = triple_to_k222s_np[t_id]
            if len(ks) > 0:
                self.k_counts[ks] -= 1
        
        def present_indices(self):
            return np.where(self.present)[0]
        
        def copy(self):
            new_state = State()
            new_state.present = self.present.copy()
            new_state.k_counts = self.k_counts.copy()
            new_state.size = self.size
            return new_state
    
    def polish(state, max_iters=100000):
        """Try to add any missing triple that fits, in order of least constraint."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Get missing triples sorted by constraint count (ascending)
            missing = [(triple_k222_count[t], t) for t in range(num_triples) if not state.present[t]]
            missing.sort()
            for _, t_id in missing:
                if state.add(t_id):
                    improved = True
            # Also try random additions
            for _ in range(500):
                if time.time() > DEADLINE:
                    break
                t_id = random.randint(0, num_triples - 1)
                if not state.present[t_id] and state.add(t_id):
                    improved = True
            # Aggressive improve: try net-positive exchanges
            if aggressive_improve(state, max_iters=2000):
                improved = True
    
    def aggressive_improve(state, max_iters=5000):
        """Try net-positive exchanges: remove 1, add 2 (or more) to grow the set."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Find K(2,2,2)s that are at count 7 (tight)
            tight_ks = np.where(state.k_counts == 7)[0]
            if len(tight_ks) == 0:
                break
            random.shuffle(tight_ks.tolist())
            for k_idx in tight_ks[:100]:
                if time.time() > DEADLINE:
                    break
                # Get the 8 triples in this K
                k_triples = k222_triples[k_idx]
                present_in_k = [t for t in k_triples if state.present[t]]
                if len(present_in_k) < 7:
                    continue
                # Try removing one present triple from this K
                random.shuffle(present_in_k)
                for t_out in present_in_k:
                    state.remove(t_out)
                    added = 0
                    # Try to add missing triples that were blocked by this K
                    missing_candidates = [t for t in range(num_triples) if not state.present[t]]
                    # Prioritize triples that are in this K or share many K's with tight ones
                    random.shuffle(missing_candidates)
                    for t_in in missing_candidates:
                        if state.add(t_in):
                            added += 1
                            if added >= 2:
                                break
                    if added > 1:
                        improved = True
                    elif added == 1:
                        # Net zero, keep it (might enable future additions)
                        improved = True
                    else:
                        # Couldn't add anything, restore
                        state.add(t_out)
                        break
                if improved:
                    break
        return improved
    
    def simulated_annealing(state, max_steps=5000):
        """
        Simulated Annealing to escape local optima.
        Moves: Add a triple, Remove a triple, Swap a triple.
        Returns (size, present_list)
        """
        current_state = state.copy()
        current_size = current_state.size
        best_state = current_state.copy()
        best_size = current_size
        
        temp = 2.0
        cooling = 0.999
        min_temp = 0.1
        
        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
            if temp < min_temp:
                break
            
            move_type = random.random()
            new_state = current_state.copy()
            accepted = False
            
            if move_type < 0.4:  # Add
                missing = [i for i in range(num_triples) if not current_state.present[i]]
                if missing:
                    t_in = random.choice(missing)
                    if new_state.add(t_in):
                        delta = 1
                        if delta >= 0 or random.random() < math.exp(-delta / temp):
                            accepted = True
                            
            elif move_type < 0.7:  # Remove
                present_list = current_state.present_indices().tolist()
                if present_list:
                    t_out = random.choice(present_list)
                    new_state.remove(t_out)
                    delta = -1
                    if delta >= 0 or random.random() < math.exp(-delta / temp):
                        accepted = True

            else:  # Swap
                present_list = current_state.present_indices().tolist()
                missing = [i for i in range(num_triples) if not current_state.present[i]]
                if present_list and missing:
                    t_out = random.choice(present_list)
                    t_in = random.choice(missing)
                    new_state.remove(t_out)
                    if new_state.add(t_in):
                        delta = 0
                        if random.random() < 0.3:
                            accepted = True

            if accepted:
                current_state = new_state
                if current_state.size > best_size:
                    best_size = current_state.size
                    best_state = current_state.copy()
                
            temp *= cooling
            
        return best_size, best_state
    
    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = State()
        # Sort by number of K(2,2,2)s (fewer constraints first) with randomness
        base_order = sorted(range(num_triples), key=lambda t: (triple_k222_count[t], local_rng.random()))
        
        for t_id in base_order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        polish(state)
        return state
    
    best_size = 0
    best_state = None
    rng = random.Random(42)
    
    # Main loop: multiple restarts
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 42 + restart_count * 7919
        
        if restart_count <= 5:
            # Greedy init
            state = greedy_init(seed)
        else:
            # Start from best and perturb
            state = State()
            if best_state:
                for i in best_state.present_indices():
                    state.add(i)
            
            # Perturb: remove some random present triples
            n_perturb = min(20, state.size // 8 + 5)
            present = state.present_indices().tolist()
            for _ in range(n_perturb):
                if present:
                    t = random.choice(present)
                    state.remove(t)
                    present = state.present_indices().tolist()
            
            polish(state)
        
        if state.size > best_size:
            best_size = state.size
            best_state = state.copy()
        
        # Simulated annealing
        sa_size, sa_state = simulated_annealing(state)
        if sa_size > best_size:
            best_size = sa_size
            best_state = sa_state.copy()
        
        # Polish after SA
        polish(sa_state)
        if sa_state.size > best_size:
            best_size = sa_state.size
            best_state = sa_state.copy()
    
    # Final high-precision polish pass
    if best_state and time.time() < DEADLINE:
        final_state = best_state.copy()
        polish(final_state, max_iters=200000)
        if final_state.size > best_size:
            best_size = final_state.size
            best_state = final_state
    
    return [list(triples[t_id]) for t_id in best_state.present_indices()]
