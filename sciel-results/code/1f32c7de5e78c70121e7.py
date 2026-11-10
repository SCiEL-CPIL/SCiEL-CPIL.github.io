import time
import random
from itertools import combinations
from math import comb

# Precompute all triples on 16 points
ALL_TRIPLES = list(combinations(range(16), 3))
TRIPLE_COUNT = len(ALL_TRIPLES)

# Precompute masks for triples
TRIPLE_MASKS = [0] * TRIPLE_COUNT
for i, t in enumerate(ALL_TRIPLES):
    mask = 0
    for x in t:
        mask |= (1 << x)
    TRIPLE_MASKS[i] = mask

# Precompute pair masks for each triple (3 pairs per triple)
TRIPLE_PAIRS = []
for t in ALL_TRIPLES:
    a, b, c = t
    TRIPLE_PAIRS.append([(a, b), (a, c), (b, c)])

# K range for independence number tracking
K_MIN = 4
K_MAX = 8
NUM_K = K_MAX - K_MIN + 1

# Precompute for each triple and each k, the list of subset masks of size k that contain the triple
# SUPERSETS[i][k_idx] = list of masks
SUPERSETS = [[[] for _ in range(NUM_K)] for _ in range(TRIPLE_COUNT)]

for i, t in enumerate(ALL_TRIPLES):
    t_set = set(t)
    remaining_points = [p for p in range(16) if p not in t_set]
    t_mask = TRIPLE_MASKS[i]
    
    for k_idx in range(NUM_K):
        k = K_MIN + k_idx
        need = k - 3
        if need < 0 or need > len(remaining_points):
            continue
        for comb_pts in combinations(remaining_points, need):
            mask = t_mask
            for p in comb_pts:
                mask |= (1 << p)
            SUPERSETS[i][k_idx].append(mask)

# Counters for tracking independence number
count_hits = [[0] * (1 << 16) for _ in range(NUM_K)]
num_independent = [0] * NUM_K

def reset_counters():
    for k_idx in range(NUM_K):
        count_hits[k_idx] = [0] * (1 << 16)
        num_independent[k_idx] = comb(16, K_MIN + k_idx)

def add_triple(i):
    for k_idx in range(NUM_K):
        for mask in SUPERSETS[i][k_idx]:
            if count_hits[k_idx][mask] == 0:
                num_independent[k_idx] -= 1
            count_hits[k_idx][mask] += 1

def remove_triple(i):
    for k_idx in range(NUM_K):
        for mask in SUPERSETS[i][k_idx]:
            count_hits[k_idx][mask] -= 1
            if count_hits[k_idx][mask] == 0:
                num_independent[k_idx] += 1

def get_alpha():
    for k_idx in range(NUM_K - 1, -1, -1):
        if num_independent[k_idx] > 0:
            return K_MIN + k_idx
    return K_MIN - 1

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_alpha = 16
    best_triples = []
    
    # Greedy initialization with multiple seeds
    def greedy_construct(seed):
        random.seed(seed)
        indices = list(range(TRIPLE_COUNT))
        random.shuffle(indices)
        triples = []
        used_pairs = set()
        
        for idx in indices:
            a, b, c = ALL_TRIPLES[idx]
            if ((a, b) not in used_pairs and 
                (a, c) not in used_pairs and 
                (b, c) not in used_pairs):
                triples.append(idx)
                used_pairs.add((a, b))
                used_pairs.add((a, c))
                used_pairs.add((b, c))
        return triples

    # Run greedy restarts
    for s in range(10):
        if time.time() > deadline:
            break
        triples = greedy_construct(s)
        reset_counters()
        for idx in triples:
            add_triple(idx)
        alpha = get_alpha()
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples.copy()

    if not best_triples:
        best_triples = greedy_construct(0)
        reset_counters()
        for idx in best_triples:
            add_triple(idx)
        best_alpha = get_alpha()

    current_triples = best_triples.copy()
    current_alpha = best_alpha
    
    # Local search with simulated annealing
    active_set = set(current_triples)
    used_pairs = set()
    for idx in current_triples:
        a, b, c = ALL_TRIPLES[idx]
        used_pairs.add((a, b))
        used_pairs.add((a, c))
        used_pairs.add((b, c))

    def is_valid(idx):
        if idx in active_set:
            return False
        a, b, c = ALL_TRIPLES[idx]
        return ((a, b) not in used_pairs and 
                (a, c) not in used_pairs and 
                (b, c) not in used_pairs)

    def add_to_state(idx):
        active_set.add(idx)
        a, b, c = ALL_TRIPLES[idx]
        used_pairs.add((a, b))
        used_pairs.add((a, c))
        used_pairs.add((b, c))
        add_triple(idx)

    def remove_from_state(idx):
        active_set.remove(idx)
        a, b, c = ALL_TRIPLES[idx]
        used_pairs.remove((a, b))
        used_pairs.remove((a, c))
        used_pairs.remove((b, c))
        remove_triple(idx)

    # Simulated annealing
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 0.01
    
    while time.time() < deadline and temperature >= min_temperature:
        move_type = random.random()
        new_alpha = current_alpha
        
        if move_type < 0.5:
            # Try to add a triple
            for _ in range(10):
                cand_idx = random.randint(0, TRIPLE_COUNT - 1)
                if is_valid(cand_idx):
                    add_to_state(cand_idx)
                    new_alpha = get_alpha()
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_triples = list(active_set)
                        break
                    elif new_alpha == current_alpha:
                        if random.random() < 0.5:
                            current_alpha = new_alpha
                            break
                        else:
                            remove_from_state(cand_idx)
                    else:
                        delta = new_alpha - current_alpha
                        prob = 0.1 if delta == 1 else 0.0
                        if random.random() < prob:
                            current_alpha = new_alpha
                            break
                        else:
                            remove_from_state(cand_idx)
                    break

        elif move_type < 0.8:
            # Try to remove a triple
            if len(active_set) > 0:
                cand_idx = random.choice(list(active_set))
                remove_from_state(cand_idx)
                new_alpha = get_alpha()
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = list(active_set)
                elif new_alpha == current_alpha:
                    if random.random() < 0.1:
                        current_alpha = new_alpha
                    else:
                        add_to_state(cand_idx)
                else:
                    add_to_state(cand_idx)

        else:
            # Swap: Remove one, add one
            if len(active_set) > 0:
                rem_idx = random.choice(list(active_set))
                remove_from_state(rem_idx)
                
                for _ in range(10):
                    add_idx = random.randint(0, TRIPLE_COUNT - 1)
                    if is_valid(add_idx):
                        add_to_state(add_idx)
                        new_alpha = get_alpha()
                        
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            if new_alpha < best_alpha:
                                best_alpha = new_alpha
                                best_triples = list(active_set)
                            break
                        elif new_alpha == current_alpha:
                            if random.random() < 0.5:
                                current_alpha = new_alpha
                                break
                            else:
                                remove_from_state(add_idx)
                                add_to_state(rem_idx)
                        else:
                            delta = new_alpha - current_alpha
                            prob = 0.1 if delta == 1 else 0.0
                            if random.random() < prob:
                                current_alpha = new_alpha
                                break
                            else:
                                remove_from_state(add_idx)
                                add_to_state(rem_idx)
                        break
                else:
                    add_to_state(rem_idx)

        temperature *= cooling_rate

    return [list(ALL_TRIPLES[i]) for i in best_triples]
