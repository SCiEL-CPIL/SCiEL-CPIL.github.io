import itertools
import time
import random
import math

V = 19
K = 7
T = 4

subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# Each block covers C(7,4)=35 subsets
all_masks = [0] * N_BLOCKS
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for i, block in enumerate(all_blocks):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(i)
    all_masks[i] = mask

block_bits = []
for i in range(N_BLOCKS):
    bits = []
    mm = all_masks[i]
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bits.append(bits)

# Rarity: number of blocks containing each subset (all 455, but useful for weighting)
subset_rarity = [len(blocks_containing_subset[s]) for s in range(N_SUBSETS)]
# Weight: inverse rarity (harder subsets get higher weight)
subset_weight = [1.0 / subset_rarity[s] for s in range(N_SUBSETS)]

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 145  # Safety margin (150s kill, 5s buffer)

    best_len = float('inf')
    best_blocks = []

    def create_cover_state():
        return {
            'blocks': [],
            'block_set': set(),
            'counts': [0] * N_SUBSETS
        }

    def add_block(state, idx):
        state['blocks'].append(idx)
        state['block_set'].add(idx)
        for bit_idx in block_bits[idx]:
            state['counts'][bit_idx] += 1

    def remove_block(state, idx):
        state['block_set'].remove(idx)
        pos = state['blocks'].index(idx)
        state['blocks'][pos] = state['blocks'][-1]
        state['blocks'].pop()
        for bit_idx in block_bits[idx]:
            state['counts'][bit_idx] -= 1

    def is_redundant(state, idx):
        for bit_idx in block_bits[idx]:
            if state['counts'][bit_idx] == 1:
                return False
        return True

    def prune_redundant(state):
        removed = 0
        changed = True
        while changed:
            changed = False
            for i in range(len(state['blocks']) - 1, -1, -1):
                idx = state['blocks'][i]
                if is_redundant(state, idx):
                    remove_block(state, idx)
                    removed += 1
                    changed = True
        return removed

    def weighted_gain(state, idx, uncov_mask):
        """Compute weighted gain of adding block idx given current uncovered mask."""
        gain = 0.0
        for bit_idx in block_bits[idx]:
            if (uncov_mask >> bit_idx) & 1:
                gain += subset_weight[bit_idx]
        return gain

    def greedy_fill_weighted(state, uncov_mask, time_limit):
        """Fill uncovered subsets using weighted greedy."""
        while uncov_mask and time.time() < time_limit:
            # Find hardest uncovered subset (highest weight)
            best_subset = -1
            best_weight = 0.0
            mm = uncov_mask
            while mm:
                lsb = mm & -mm
                s = lsb.bit_length() - 1
                if subset_weight[s] > best_weight:
                    best_weight = subset_weight[s]
                    best_subset = s
                mm ^= lsb
            
            if best_subset == -1:
                break
            
            # Find best block covering best_subset
            candidates = blocks_containing_subset[best_subset]
            best_idx = -1
            best_gain = 0.0
            for c in candidates:
                if c in state['block_set']:
                    continue
                gain = weighted_gain(state, c, uncov_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = c
            
            if best_idx == -1:
                # No available block covers this subset; break
                break
            
            add_block(state, best_idx)
            uncov_mask &= ~all_masks[best_idx]
        
        return uncov_mask

    def try_swap(state, time_limit):
        """Try 1-for-1 swap."""
        n_blocks = len(state['blocks'])
        if n_blocks < 2:
            return False
        
        max_attempts = min(30, n_blocks)
        for _ in range(max_attempts):
            if time.time() > time_limit:
                return False
            i = random.randint(0, n_blocks - 1)
            idx = state['blocks'][i]
            
            unique_bits = []
            for bit_idx in block_bits[idx]:
                if state['counts'][bit_idx] == 1:
                    unique_bits.append(bit_idx)
            
            if not unique_bits:
                remove_block(state, idx)
                return True
            
            if len(unique_bits) <= 10:
                unique_bits.sort(key=lambda b: subset_rarity[b])
                
                candidates = set(blocks_containing_subset[unique_bits[0]])
                for b in unique_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                
                valid_candidates = [c for c in candidates if c not in state['block_set']]
                
                if valid_candidates:
                    # Pick candidate with best weighted gain
                    best_new = valid_candidates[0]
                    best_gain = -1
                    for c in valid_candidates:
                        gain = 0.0
                        for bit_idx in block_bits[c]:
                            if state['counts'][bit_idx] == 0:
                                gain += subset_weight[bit_idx]
                        if gain > best_gain:
                            best_gain = gain
                            best_new = c
                    
                    remove_block(state, idx)
                    add_block(state, best_new)
                    return True
        
        return False

    def try_swap_2_for_1(state, time_limit):
        """Try to replace 2 blocks with 1 block."""
        n_blocks = len(state['blocks'])
        if n_blocks < 2:
            return False
        
        max_attempts = 25
        for _ in range(max_attempts):
            if time.time() > time_limit:
                return False
            
            i, j = random.sample(range(n_blocks), 2)
            idx1 = state['blocks'][i]
            idx2 = state['blocks'][j]
            
            candidates_unique = set()
            for s in block_bits[idx1]:
                c = state['counts'][s]
                if c == 1:
                    candidates_unique.add(s)
                elif c == 2 and (all_masks[idx2] >> s) & 1:
                    candidates_unique.add(s)
            
            for s in block_bits[idx2]:
                c = state['counts'][s]
                if c == 1:
                    candidates_unique.add(s)
                elif c == 2 and (all_masks[idx1] >> s) & 1:
                    candidates_unique.add(s)
            
            if not candidates_unique:
                continue
            if len(candidates_unique) > 12:
                continue
            
            cand_list = list(candidates_unique)
            cand_list.sort(key=lambda b: subset_rarity[b])
            
            common_candidates = set(blocks_containing_subset[cand_list[0]])
            for s in cand_list[1:]:
                if not common_candidates:
                    break
                common_candidates &= set(blocks_containing_subset[s])
            
            valid_candidates = [c for c in common_candidates if c not in state['block_set']]
            
            if valid_candidates:
                new_idx = random.choice(valid_candidates)
                remove_block(state, idx1)
                remove_block(state, idx2)
                add_block(state, new_idx)
                return True
        
        return False

    def local_search(state, time_limit):
        """Run local search: prune, swap, 2-for-1, perturb."""
        no_improve = 0
        max_no_improve = 40
        
        while time.time() < time_limit:
            if prune_redundant(state) > 0:
                no_improve = 0
                continue
            
            if try_swap(state, time_limit):
                no_improve = 0
                continue
            
            if try_swap_2_for_1(state, time_limit):
                no_improve = 0
                continue
            
            no_improve += 1
            if no_improve > max_no_improve:
                # Perturbation
                n_remove = random.randint(3, 7)
                to_remove = random.sample(state['blocks'], min(n_remove, len(state['blocks'])))
                for idx in to_remove:
                    remove_block(state, idx)
                
                uncov_mask = 0
                for s in range(N_SUBSETS):
                    if state['counts'][s] == 0:
                        uncov_mask |= 1 << s
                
                greedy_fill_weighted(state, uncov_mask, time_limit)
                prune_redundant(state)
                
                no_improve = 0
                max_no_improve = min(150, max_no_improve * 2)

    # Strategy 1: Multiple probabilistic initializations with weighted greedy fill
    p_values = [0.004, 0.006, 0.008, 0.010, 0.013, 0.016, 0.020, 0.025, 0.030]
    
    # Shuffle p values for diversity
    random.shuffle(p_values)
    
    for p in p_values:
        if time.time() > deadline - 25:
            break
        
        state = create_cover_state()
        
        # Randomly select blocks with probability p
        for i in range(N_BLOCKS):
            if random.random() < p:
                add_block(state, i)
        
        # Check uncovered
        uncov_mask = 0
        for s in range(N_SUBSETS):
            if state['counts'][s] == 0:
                uncov_mask |= 1 << s
        
        if uncov_mask:
            greedy_fill_weighted(state, uncov_mask, deadline - 15)
        
        # Prune
        prune_redundant(state)
        
        # Local search
        local_search(state, deadline - 10)
        
        # Update best
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()
        
        # Early exit if excellent
        if best_len <= 143:
            break

    # Strategy 2: Pure weighted greedy initialization
    if time.time() < deadline - 20:
        state = create_cover_state()
        uncov_mask = (1 << N_SUBSETS) - 1
        greedy_fill_weighted(state, uncov_mask, deadline - 10)
        prune_redundant(state)
        local_search(state, deadline - 5)
        
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()

    # Strategy 3: Another probabilistic with different p
    if time.time() < deadline - 15:
        p = 0.012
        state = create_cover_state()
        for i in range(N_BLOCKS):
            if random.random() < p:
                add_block(state, i)
        
        uncov_mask = 0
        for s in range(N_SUBSETS):
            if state['counts'][s] == 0:
                uncov_mask |= 1 << s
        
        if uncov_mask:
            greedy_fill_weighted(state, uncov_mask, deadline - 8)
        
        prune_redundant(state)
        local_search(state, deadline - 3)
        
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()

    # Final cleanup: aggressive prune
    if best_blocks:
        state = create_cover_state()
        for idx in best_blocks:
            add_block(state, idx)
        prune_redundant(state)
        if len(state['blocks']) < best_len:
            best_len = len(state['blocks'])
            best_blocks = state['blocks'].copy()

    return [list(all_blocks[i]) for i in best_blocks]
