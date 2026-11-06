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
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True
        
        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1
        
        def present_list(self):
            return [i for i in range(num_triples) if self.present[i]]
    
    def polish(state):
        """Systematically try to add all possible missing triples."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in range(num_triples):
                if not state.present[t_id]:
                    if state.add(t_id):
                        improved = True
    
    def polish_full(state):
        """Systematically try to add all possible missing triples."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in range(num_triples):
                if not state.present[t_id]:
                    if state.add(t_id):
                        improved = True
    
    def random_1swap(state):
        """Remove one present, add one missing."""
        present = state.present_list()
        if not present:
            return False
        t_out = random.choice(present)
        # Find a missing triple
        for _ in range(20):
            t_in = random.randint(0, num_triples - 1)
            if not state.present[t_in]:
                break
        else:
            return False
        
        state.remove(t_out)
        if state.add(t_in):
            return True
        else:
            state.add(t_out)
            return False
    
    def random_2swap(state):
        """Remove two present, add two missing."""
        present = state.present_list()
        if len(present) < 2:
            return False
        t_out1, t_out2 = random.sample(present, 2)
        
        missing = []
        for _ in range(40):
            t = random.randint(0, num_triples - 1)
            if not state.present[t]:
                missing.append(t)
            if len(missing) >= 2:
                break
        if len(missing) < 2:
            return False
        t_in1, t_in2 = missing
        
        state.remove(t_out1)
        state.remove(t_out2)
        added1 = state.add(t_in1)
        added2 = state.add(t_in2)
        if added1 and added2:
            return True
        else:
            # Revert
            if added1:
                state.remove(t_in1)
            if added2:
                state.remove(t_in2)
            state.add(t_out2)
            state.add(t_out1)
            return False
    
    best_size = 0
    best_present = None
    rng = random.Random(42)
    
    # Greedy initialization by K(2,2,2) count (least constrained first)
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
    
    # Main loop: multiple restarts
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 42 + restart_count * 7919
        
        if restart_count <= 5:
            # Greedy init
            state = greedy_init(seed)
        else:
            # Perturb best solution
            state = State()
            for i in range(num_triples):
                if best_present[i]:
                    state.present[i] = True
                    state.size += 1
            for k_idx in range(num_k222s):
                for t_id in k222_triples[k_idx]:
                    if state.present[t_id]:
                        state.k_counts[k_idx] += 1
            # Perturb: remove and re-add some
            n_perturb = min(20, state.size // 10)
            present = state.present_list()
            for _ in range(n_perturb):
                if present:
                    t = random.choice(present)
                    state.remove(t)
                    present = state.present_list()
            polish(state)
        
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
        
        # Local search: Iterated 1-for-2 swaps and 1-for-1 diversification
        # We use a simple temperature scheme
        temperature = 1.0
        cooling = 0.995
        min_temp = 0.05
        
        for step in range(2000):
            if time.time() > DEADLINE:
                break
            
            # 1. Try to add a single triple (Greedy)
            if random.random() < 0.3:
                # Systematic polish is better, but random is faster per step
                # Let's do a mini-polish
                t_id = random.randint(0, num_triples - 1)
                if not state.present[t_id] and state.add(t_id):
                    if state.size > best_size:
                        best_size = state.size
                        best_present = state.present[:]
                    continue

            # 2. Try 1-for-2 swap (Remove 1, Add 2) -> Size +1
            present = state.present_list()
            if not present:
                break
            t_out = random.choice(present)
            
            # Pick 2 random missing triples
            t_in1 = -1
            t_in2 = -1
            for _ in range(100):
                t = random.randint(0, num_triples - 1)
                if not state.present[t]:
                    if t_in1 == -1:
                        t_in1 = t
                    elif t_in2 == -1 and t != t_in1:
                        t_in2 = t
                        break
            if t_in1 == -1 or t_in2 == -1:
                continue
                
            state.remove(t_out)
            added1 = state.add(t_in1)
            added2 = state.add(t_in2) if added1 else False
            
            if added1 and added2:
                if state.size > best_size:
                    best_size = state.size
                    best_present = state.present[:]
            else:
                if added1:
                    state.remove(t_in1)
                if added2:
                    state.remove(t_in2)
                state.add(t_out)
                
            # 3. Try 1-for-1 swap (Remove 1, Add 1) -> Size 0
            # Accept with probability exp(0/T) = 1? 
            # To avoid random walk, only accept if it "helps"? 
            # Hard to define help. Let's accept with prob 0.5 to diversify.
            if random.random() < 0.5:
                t_out = random.choice(state.present_list())
                t_in = -1
                for _ in range(50):
                    t = random.randint(0, num_triples - 1)
                    if not state.present[t]:
                        t_in = t
                        break
                if t_in != -1:
                    state.remove(t_out)
                    if state.add(t_in):
                        pass
                    else:
                        state.add(t_out)
                        
            temperature *= cooling
            if temperature < min_temp:
                temperature = min_temp
                
        # Final systematic polish
        polish(state)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
    
    return [list(triples[t_id]) for t_id in range(num_triples) if best_present[t_id]]
