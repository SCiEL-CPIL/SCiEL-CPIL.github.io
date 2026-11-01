import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 0:
        return [[]]
    
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Generic fallback
    return _generic_construct(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    q, n, r = 3, 6, 1
    total = q ** n  # 729

    # Precompute all words as indices
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbor lists and bitsets
    # neighbors_list[i] is a list of indices
    # neighbors_bit[i] is an integer where bit j is 1 if j is a neighbor of i
    neighbors_list = [None] * total
    neighbors_bit = [0] * total
    
    for i, w in enumerate(all_words):
        nb_set = {i}
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                nb_set.add(idx_map[tuple(nw)])
        neighbors_list[i] = list(nb_set)
        # Create bitset
        mask = 0
        for nb in nb_set:
            mask |= (1 << nb)
        neighbors_bit[i] = mask

    # Initial code: Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code_set = set()
    for head in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = head + list(tail)
            code_set.add(idx_map[tuple(word)])

    # Coverage counts
    cover_count = [0] * total
    for c in code_set:
        for nb in neighbors_list[c]:
            cover_count[nb] += 1

    # Helper to check validity
    def is_valid():
        # Check if any cover_count is 0
        return min(cover_count) > 0

    # Local Search
    # We use bitsets to speed up checks.
    # critical_mask: bitset of words with cover_count == 1
    # If we remove c, words in (neighbors_bit[c] & critical_mask) become uncovered.
    # If we add new_c, it covers neighbors_bit[new_c].
    
    def local_search(code_set, cc, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Try removing codewords
            # A codeword c can be removed if no word is covered ONLY by c.
            # i.e., for all nb in neighbors(c), cc[nb] > 1.
            for c in list(code_set):
                if time.time() > deadline:
                    return
                can_remove = True
                for nb in neighbors_list[c]:
                    if cc[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    code_set.remove(c)
                    for nb in neighbors_list[c]:
                        cc[nb] -= 1
                    improved = True

            if time.time() > deadline:
                return

            # Pass 2: Try 2-for-1 swaps
            # Remove c1, c2. Add new_c.
            # Condition: 
            # 1. new_c not in code_set
            # 2. All words uniquely covered by c1 or c2 must be covered by new_c.
            # Let U1 = {nb | cc[nb]==1 and nb in N(c1)}
            # Let U2 = {nb | cc[nb]==1 and nb in N(c2)}
            # Note: If a word is in U1 and U2, it's covered by both, so cc[nb] would be 2, not 1.
            # So U1 and U2 are disjoint if we consider the state BEFORE removal.
            # Actually, if nb in N(c1) and N(c2), cc[nb] >= 2. So unique sets are disjoint.
            # The set of words that would become uncovered is U1 U U2.
            # We need N(new_c) to contain U1 U U2.
            
            code_list = list(code_set)
            random.shuffle(code_list)
            
            # To make this efficient, we only try pairs if the code size is large enough
            # and we don't try all pairs (O(N^2) is too slow for 70+ words in Python loops).
            # We sample pairs.
            if len(code_list) > 2:
                for i in range(len(code_list) - 1):
                    if time.time() > deadline:
                        return
                    c1 = code_list[i]
                    c2 = code_list[i+1]
                    if c2 not in code_set: continue # might have been removed? No, list is snapshot
                    
                    # Calculate unique words for c1 and c2
                    # unique_mask = bitset of words with cc==1 in N(c1) U N(c2)
                    unique_mask = 0
                    for nb in neighbors_list[c1]:
                        if cc[nb] == 1:
                            unique_mask |= (1 << nb)
                    for nb in neighbors_list[c2]:
                        if cc[nb] == 1:
                            unique_mask |= (1 << nb)
                            
                    if unique_mask == 0:
                        # If no unique words, we can just remove one of them? 
                        # Pass 1 should have caught single removals. 
                        # But maybe removing both allows a better state? 
                        # Actually, if unique_mask is 0, removing c1 is valid (Pass 1). 
                        # So this case is handled by Pass 1.
                        continue

                    # Find candidates new_c that cover unique_mask
                    # new_c must satisfy: (neighbors_bit[new_c] & unique_mask) == unique_mask
                    # And new_c not in code_set.
                    
                    # To speed up, we can precompute for each possible unique_mask? No, masks vary.
                    # We iterate through all possible new_c not in code_set.
                    # 729 candidates max.
                    # Check: (neighbors_bit[new_c] & unique_mask) == unique_mask
                    
                    found = False
                    # Randomize candidate order to escape local minima
                    candidates = [c for c in range(total) if c not in code_set]
                    random.shuffle(candidates)
                    
                    for new_c in candidates:
                        if (neighbors_bit[new_c] & unique_mask) == unique_mask:
                            # Check if adding new_c and removing c1, c2 keeps everything else covered.
                            # Words covered by c1 or c2 but NOT by new_c must be covered by other codewords.
                            # i.e., for nb in (N(c1) U N(c2)) \ N(new_c), cc[nb] - 1 (for c1) - 1 (for c2) + 1 (for new_c if covered) >= 1?
                            # Actually, simpler:
                            # New cc' for nb:
                            # old_cc = cc[nb]
                            # delta = -1 (if nb in N(c1)) -1 (if nb in N(c2)) + 1 (if nb in N(new_c))
                            # We need old_cc + delta >= 1.
                            # i.e., old_cc >= 1 - delta.
                            
                            # Since unique_mask is covered by new_c, for nb in unique_mask:
                            # old_cc = 1.
                            # delta = -1 -1 + 1 = -1 (if nb in N(c1) and N(c2)? No, unique implies only one of c1,c2 covers it uniquely? 
                            # Wait. If nb in N(c1) and cc[nb]==1, then no other codeword covers nb. 
                            # So nb is NOT in N(c2) unless c2==c1 (impossible). 
                            # So for nb in unique_mask, it is in N(c1) XOR N(c2). 
                            # So delta = -1 + 1 = 0. 
                            # So cc'[nb] = 1. OK.
                            
                            # For nb NOT in unique_mask:
                            # If nb in N(c1) and cc[nb] > 1:
                            #   If nb in N(c2), delta = -1 -1 + (1 if nb in N(new_c) else 0).
                            #   If nb not in N(c2), delta = -1 + (1 if nb in N(new_c) else 0).
                            # We need to check all nb in N(c1) U N(c2).
                            
                            # Optimization: 
                            # The only words that might become uncovered are those in (N(c1) U N(c2)) \ N(new_c).
                            # For these words, we lose coverage from c1 and/or c2.
                            # We need to ensure they are covered by at least one OTHER codeword.
                            # i.e., cc[nb] - (1 if nb in N(c1) else 0) - (1 if nb in N(c2) else 0) >= 1.
                            
                            valid_swap = True
                            # Iterate neighbors of c1 and c2
                            # To avoid double checking, use a set or just check condition carefully.
                            checked = set()
                            for nb in neighbors_list[c1]:
                                if nb in checked: continue
                                checked.add(nb)
                                loss = 1
                                if nb in neighbors_list[c2]: loss += 1
                                if nb in neighbors_list[new_c]: loss -= 1
                                # Actually, we just need to check if cc[nb] - loss >= 1
                                # cc[nb] is current count.
                                if cc[nb] - loss < 1:
                                    valid_swap = False
                                    break
                            if valid_swap:
                                for nb in neighbors_list[c2]:
                                    if nb in checked: continue
                                    checked.add(nb)
                                    loss = 1
                                    if nb in neighbors_list[new_c]: loss -= 1
                                    if cc[nb] - loss < 1:
                                        valid_swap = False
                                        break
                            
                            if valid_swap:
                                # Perform swap
                                code_set.remove(c1)
                                code_set.remove(c2)
                                code_set.add(new_c)
                                for nb in neighbors_list[c1]:
                                    cc[nb] -= 1
                                for nb in neighbors_list[c2]:
                                    cc[nb] -= 1
                                for nb in neighbors_list[new_c]:
                                    cc[nb] += 1
                                improved = True
                                found = True
                                break
                    if found:
                        # Restart local search loop to catch new removals
                        break

    def greedy_repair(code_set, cc, deadline):
        """Add codewords to cover all uncovered words."""
        while min(cc) < 1 and time.time() < deadline:
            uncovered = [i for i in range(total) if cc[i] == 0]
            if not uncovered:
                break
            uncovered_set = set(uncovered)
            best_c = -1
            best_cnt = -1
            for c in range(total):
                if c in code_set:
                    continue
                cnt = 0
                for nb in neighbors_list[c]:
                    if nb in uncovered_set:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt == 0:
                break
            code_set.add(best_c)
            for nb in neighbors_list[best_c]:
                cc[nb] += 1

    # Run initial local search
    local_search(code_set, cover_count, deadline)
    greedy_repair(code_set, cover_count, deadline)
    
    best_size = len(code_set)
    best_code = set(code_set)
    best_cc = list(cover_count)

    # ILS
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        # Perturb
        k = random.randint(2, 5)
        code_list = list(code_set)
        random.shuffle(code_list)
        removed = code_list[:k]
        for c in removed:
            code_set.remove(c)
            for nb in neighbors_list[c]:
                cover_count[nb] -= 1
        
        greedy_repair(code_set, cover_count, deadline)
        local_search(code_set, cover_count, deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            best_cc = list(cover_count)
            if best_size <= 73:
                break

    code_set = best_code
    cover_count = best_cc
    
    # Final polish
    local_search(code_set, cover_count, deadline)
    greedy_repair(code_set, cover_count, deadline)
    
    return [list(all_words[c]) for c in sorted(code_set)]


def _generic_construct(q, n, radius, deadline):
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        for r2 in range(radius + 1):
            for pos in itertools.combinations(range(n), r2):
                for vals in itertools.product(range(q), repeat=r2):
                    nw = list(w)
                    for p, v in zip(pos, vals):
                        nw[p] = v
                    neighbors[i].add(idx_map[tuple(nw)])
    
    neighbors_list = [list(s) for s in neighbors]
    
    code = set()
    cover_count = [0] * total
    
    while time.time() < deadline:
        uncovered = [i for i in range(total) if cover_count[i] == 0]
        if not uncovered:
            break
        
        best_c = -1
        best_cnt = -1
        for c in range(total):
            if c in code:
                continue
            cnt = sum(1 for u in uncovered if u in neighbors[c])
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c
        
        if best_c == -1 or best_cnt == 0:
            break
        
        code.add(best_c)
        for nb in neighbors_list[best_c]:
            cover_count[nb] += 1
    
    return [list(all_words[c]) for c in sorted(code)]
