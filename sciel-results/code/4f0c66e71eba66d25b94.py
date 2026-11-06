import random
import time
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all K(2,2,2) structures
    structs = []
    struct_to_triples = []
    triple_to_structs = [[] for _ in range(num_triples)]
    
    def get_idx(a, b, c):
        return triple_to_idx[tuple(sorted((a, b, c)))]
        
    for six_pts in combinations(range(n), 6):
        p = list(six_pts)
        pairings = []
        def gen_pairings(remaining, current):
            if not remaining:
                pairings.append(tuple(sorted(current, key=lambda x: min(x))))
                return
            first = remaining[0]
            for i in range(1, len(remaining)):
                second = remaining[i]
                new_remaining = remaining[1:i] + remaining[i+1:]
                gen_pairings(new_remaining, current + [(first, second)])
        
        gen_pairings(p, [])
        
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_struct = []
            for x1 in p1:
                for x2 in p2:
                    for x3 in p3:
                        idx = get_idx(x1, x2, x3)
                        triples_in_struct.append(idx)
            
            struct_id = len(struct_to_triples)
            struct_to_triples.append(triples_in_struct)
            for t_idx in triples_in_struct:
                triple_to_structs[t_idx].append(struct_id)
                
    num_structs = len(struct_to_triples)
    
    in_set = [False] * num_triples
    counts = [0] * num_structs
    current_size = 0
    best_size = 0
    best_solution = []
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin
    
    def add_triple(idx):
        nonlocal current_size
        in_set[idx] = True
        current_size += 1
        for s_idx in triple_to_structs[idx]:
            counts[s_idx] += 1
            
    def remove_triple(idx):
        nonlocal current_size
        in_set[idx] = False
        current_size -= 1
        for s_idx in triple_to_structs[idx]:
            counts[s_idx] -= 1
            
    def is_valid_add(idx):
        if in_set[idx]:
            return False
        for s_idx in triple_to_structs[idx]:
            if counts[s_idx] == 7:
                return False
        return True
        
    def is_valid_swap(in_idx, out_idx):
        # Check if swapping in_idx (remove) and out_idx (add) is valid
        # in_idx is currently in set, out_idx is not.
        if in_set[in_idx] and not in_set[out_idx]:
            in_structs_set = set(triple_to_structs[in_idx])
            for s_idx in triple_to_structs[out_idx]:
                if s_idx not in in_structs_set:
                    if counts[s_idx] >= 7:
                        return False
            return True
        return False

    def greedy_init_return_list(seed):
        in_set_local = [False] * num_triples
        counts_local = [0] * num_structs
        current_size_local = 0
        
        def is_valid_add_local(idx):
            if in_set_local[idx]:
                return False
            for s_idx in triple_to_structs[idx]:
                if counts_local[s_idx] == 7:
                    return False
            return True
            
        def add_triple_local(idx):
            nonlocal current_size_local
            in_set_local[idx] = True
            current_size_local += 1
            for s_idx in triple_to_structs[idx]:
                counts_local[s_idx] += 1
                
        random.seed(seed)
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            if is_valid_add_local(idx):
                add_triple_local(idx)
                
        return [i for i in range(num_triples) if in_set_local[i]]

    # Run multiple greedy restarts
    best_init_list = []
    best_init_size = 0
    for seed in range(20):
        init_list = greedy_init_return_list(seed)
        if len(init_list) > best_init_size:
            best_init_size = len(init_list)
            best_init_list = init_list

    # Load best init into global state
    in_set = [False] * num_triples
    counts = [0] * num_structs
    current_size = 0
    for idx in best_init_list:
        add_triple(idx)
        
    best_size = current_size
    best_solution = [list(all_triples[i]) for i in best_init_list]
    
    # Local Search
    while time.time() - start_time < time_limit:
        # Phase 1: Add
        added_any = True
        while added_any and (time.time() - start_time < time_limit):
            added_any = False
            order = list(range(num_triples))
            random.shuffle(order)
            for idx in order:
                if not in_set[idx] and is_valid_add(idx):
                    add_triple(idx)
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
        
        # If we added nothing, we are stuck.
        if not added_any:
            if time.time() - start_time > time_limit - 10:
                break
                
            # Phase 2: Swap
            swaps_done = 0
            max_swaps = 10
            in_list = [i for i in range(num_triples) if in_set[i]]
            out_list = [i for i in range(num_triples) if not in_set[i]]
            
            if len(in_list) > 0 and len(out_list) > 0:
                for _ in range(max_swaps):
                    if time.time() - start_time > time_limit - 10:
                        break
                    in_idx = random.choice(in_list)
                    out_idx = random.choice(out_list)
                    
                    if is_valid_swap(in_idx, out_idx):
                        remove_triple(in_idx)
                        add_triple(out_idx)
                        in_list.remove(in_idx)
                        out_list.remove(out_idx)
                        in_list.append(out_idx)
                        out_list.append(in_idx)
                        swaps_done += 1
                        
            # Phase 3: Try adding again after swaps
            added_after_swap = False
            order = list(range(num_triples))
            random.shuffle(order)
            for idx in order:
                if not in_set[idx] and is_valid_add(idx):
                    add_triple(idx)
                    added_after_swap = True
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]
            
            if not added_after_swap:
                # Kick: Remove k random triples
                k = random.randint(3, 8)
                current_indices = [i for i in range(num_triples) if in_set[i]]
                if len(current_indices) >= k:
                    to_remove = random.sample(current_indices, k)
                    for idx in to_remove:
                        remove_triple(idx)
                    # Re-greedy fill
                    order = list(range(num_triples))
                    random.shuffle(order)
                    for idx in order:
                        if not in_set[idx] and is_valid_add(idx):
                            add_triple(idx)
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = [list(all_triples[i]) for i in range(num_triples) if in_set[i]]

    return best_solution
