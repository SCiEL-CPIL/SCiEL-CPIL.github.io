import time
import random
import math
import itertools

def _precompute_neighbors(q, n, radius):
    """Precompute radius-1 neighbors for covering, and distance-2 pairs for conflict detection."""
    total = q ** n
    neigh = [[] for _ in range(total)]
    neigh_set = [set() for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        if radius >= 1:
            for i in range(n):
                pow_q_i = q ** i
                digit = (idx // pow_q_i) % q
                for new_digit in range(q):
                    if new_digit != digit:
                        new_idx = idx + (new_digit - digit) * pow_q_i
                        neigh[idx].append(new_idx)
                        neigh_set[idx].add(new_idx)
    
    # Precompute distance-2 conflict graph (for independent set constraint)
    conflict = [set() for _ in range(total)]
    if radius >= 1:
        for i in range(n):
            pow_q_i = q ** i
            for j in range(n):
                if i == j:
                    continue
                pow_q_j = q ** j
                for idx in range(total):
                    di = (idx // pow_q_i) % q
                    dj = (idx // pow_q_j) % q
                    for ndi in range(q):
                        if ndi == di:
                            continue
                        for ndj in range(q):
                            if ndj == dj:
                                continue
                            nidx = idx + (ndi - di) * pow_q_i + (ndj - dj) * pow_q_j
                            conflict[idx].add(nidx)
    
    return neigh, neigh_set, conflict, total

def _greedy_independent_set_init(q, n, radius, neigh, neigh_set, conflict, total, rng):
    """
    Greedy initialization: build a maximal independent set in the distance-2 graph
    while ensuring coverage. Start with a covering set, then greedily add words
    that don't conflict (distance 2) with existing set members, prioritizing
    those that cover the most uncovered words.
    """
    uncovered = [True] * total
    rem_count = total
    code = []
    code_set = set()
    
    # First pass: greedy set cover
    while rem_count > 0:
        best_c = -1
        best_gain = 0
        for c in range(total):
            if c in code_set:
                continue
            gain = 0
            for u in neigh[c]:
                if uncovered[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain == 0:
            break
        code.append(best_c)
        code_set.add(best_c)
        for u in neigh[best_c]:
            if uncovered[u]:
                uncovered[u] = False
                rem_count -= 1
    
    # Second pass: try to replace pairs at distance 2 with single better words
    # This breaks conflict clauses
    changed = True
    while changed and rem_count == 0:
        changed = False
        for i in range(len(code)):
            for j in range(i + 1, len(code)):
                if code[j] in conflict[code[i]]:
                    # These two are at distance 2; try to find a single word
                    # that covers the union of their unique coverage
                    unique_i = set()
                    unique_j = set()
                    for u in neigh[code[i]]:
                        if u not in neigh_set[code[j]]:
                            unique_i.add(u)
                    for u in neigh[code[j]]:
                        if u not in neigh_set[code[i]]:
                            unique_j.add(u)
                    
                    # Try to find a single word covering both unique sets
                    best_repl = -1
                    best_miss = total + 1
                    for c in range(total):
                        if c in code_set or c == code[i] or c == code[j]:
                            continue
                        miss = 0
                        for u in unique_i | unique_j:
                            if u not in neigh_set[c]:
                                miss += 1
                        if miss < best_miss:
                            best_miss = miss
                            best_repl = c
                    
                    if best_repl != -1 and best_miss == 0:
                        # Perfect replacement
                        code[i] = best_repl
                        code_set.discard(code[i])
                        code_set.add(best_repl)
                        code.pop(j)
                        code_set.discard(code[j])
                        changed = True
                        break
            if changed:
                break
    
    return code, code_set

def _clause_learning_local_search(q, n, radius, neigh, neigh_set, conflict, total, code, code_set, deadline, rng):
    """
    Local search with clause learning: maintain a set of 'clauses' that are
    violated (uncovered words) or conflicted (pairs at distance 2).
    Moves are evaluated by their impact on these clauses.
    """
    best_code = code[:]
    best_size = len(code)
    
    # Track coverage counts and conflict status
    counts = [0] * total
    for c in code:
        for u in neigh[c]:
            counts[u] += 1
    
    # Track which words in the code have conflicts
    def get_conflicts(c, code_set):
        """Return set of code members that conflict with c (distance 2)."""
        confs = set()
        for other in code_set:
            if other == c:
                continue
            if other in conflict[c]:
                confs.add(other)
        return confs
    
    # Precompute conflict membership for faster lookup
    # conflict_adj[c] = set of all words at distance 2 from c
    # We already have conflict as a list of sets
    
    temp = 1.0
    cool_rate = 0.9997
    min_temp = 0.005
    
    # Clause-based scoring: a move is good if it reduces the number of
    # uncovered words or resolves conflicts
    def move_score_remove(c):
        """Score of removing c: negative if it breaks coverage, positive if it resolves conflicts."""
        coverage_loss = 0
        for u in neigh[c]:
            if counts[u] == 1:
                coverage_loss += 1
        conflict_gain = len(get_conflicts(c, code_set))
        return conflict_gain * 10 - coverage_loss * 100  # Heavily penalize coverage loss
    
    def move_score_swap(old_c, new_c):
        """Score of swapping old_c for new_c."""
        # Coverage impact
        cov_change = 0
        for u in neigh[old_c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                cov_change -= 1  # lose coverage
        for u in neigh[new_c]:
            if u not in neigh_set[old_c]:
                cov_change += 1  # gain coverage (but may be redundant)
        
        # Conflict impact: new_c must not conflict with any remaining code member
        new_conflicts = 0
        for other in code_set:
            if other == old_c:
                continue
            if other in conflict[new_c]:
                new_conflicts += 1
        old_conflicts = len(get_conflicts(old_c, code_set))
        
        conflict_change = old_conflicts - new_conflicts
        
        return conflict_change * 10 - (-cov_change) * 100
    
    while time.monotonic() < deadline:
        if temp < min_temp:
            temp = min_temp
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try Remove
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                c = code[idx]
                # Check if removable (no unique coverage)
                removable = True
                for u in neigh[c]:
                    if counts[u] == 1:
                        removable = False
                        break
                if removable:
                    code.pop(idx)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
        
        elif move_type < 0.85:
            # Try Swap with conflict-awareness
            if len(code) > 0:
                idx = rng.randint(0, len(code) - 1)
                old_c = code[idx]
                new_c = rng.randint(0, total - 1)
                
                if new_c == old_c or new_c in code_set:
                    continue
                
                # Check coverage: no unique coverage lost
                valid = True
                for u in neigh[old_c]:
                    if counts[u] == 1 and u not in neigh_set[new_c]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check conflicts: new_c should not add too many new conflicts
                new_conflict_count = 0
                for other in code_set:
                    if other == old_c:
                        continue
                    if other in conflict[new_c]:
                        new_conflict_count += 1
                
                old_conflict_count = len(get_conflicts(old_c, code_set))
                
                # Accept if it doesn't increase conflicts too much
                if new_conflict_count <= old_conflict_count + 1:
                    for u in neigh[old_c]:
                        counts[u] -= 1
                    for u in neigh[new_c]:
                        counts[u] += 1
                    code[idx] = new_c
                    code_set.discard(old_c)
                    code_set.add(new_c)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
        
        else:
            # Perturbation: add a word that helps with coverage or resolves conflicts
            new_c = rng.randint(0, total - 1)
            if new_c not in code_set:
                if rng.random() < 0.15:
                    code.append(new_c)
                    code_set.add(new_c)
                    for u in neigh[new_c]:
                        counts[u] += 1
        
        temp *= cool_rate
    
    return best_code, best_size

def _conflict_resolution_pass(q, n, radius, neigh, neigh_set, conflict, total, code, code_set, deadline, rng):
    """
    Dedicated pass to resolve distance-2 conflicts by finding replacements.
    This is the 'clause learning' step: for each conflict clause (pair at dist 2),
    learn a replacement that satisfies both coverage and independence.
    """
    best_code = code[:]
    best_size = len(code)
    
    # Find all conflict pairs
    conflict_pairs = []
    code_list = code[:]
    for i in range(len(code_list)):
        for j in range(i + 1, len(code_list)):
            if code_list[j] in conflict[code_list[i]]:
                conflict_pairs.append((i, j))
    
    if not conflict_pairs:
        return best_code, best_size
    
    # For each conflict pair, try to resolve
    for _attempt in range(50):
        if time.monotonic() > deadline:
            break
        
        if not conflict_pairs:
            break
        
        # Pick a random conflict pair
        pi, pj = rng.choice(conflict_pairs)
        if pi >= len(code) or pj >= len(code):
            conflict_pairs = []
            continue
        
        ci = code[pi]
        cj = code[pj]
        
        # Try to replace one or both with a single word
        # Compute unique coverage of ci and cj
        unique_i = set()
        unique_j = set()
        for u in neigh[ci]:
            if u not in neigh_set[cj]:
                unique_i.add(u)
        for u in neigh[cj]:
            if u not in neigh_set[ci]:
                unique_j.add(u)
        
        union_unique = unique_i | unique_j
        
        # Try single replacement
        best_repl = -1
        best_miss = len(union_unique) + 1
        for c in range(total):
            if c in code_set or c == ci or c == cj:
                continue
            miss = 0
            for u in union_unique:
                if u not in neigh_set[c]:
                    miss += 1
                    if miss >= best_miss:
                        break
            if miss < best_miss:
                best_miss = miss
                best_repl = c
            if miss == 0:
                break
        
        if best_repl != -1 and best_miss == 0:
            # Replace both with one
            code[pi] = best_repl
            code_set.discard(ci)
            code_set.discard(cj)
            code_set.add(best_repl)
            code.pop(pj)
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
            # Recompute conflict pairs
            conflict_pairs = []
            for i in range(len(code)):
                for j in range(i + 1, len(code)):
                    if code[j] in conflict[code[i]]:
                        conflict_pairs.append((i, j))
        
        else:
            # Try replacing just one
            for idx, c_to_replace in [(pi, ci), (pj, cj)]:
                other = cj if c_to_replace == ci else ci
                unique_to_replace = unique_i if c_to_replace == ci else unique_j
                
                best_single = -1
                best_single_miss = len(unique_to_replace) + 1
                for c in range(total):
                    if c in code_set or c == c_to_replace or c == other:
                        continue
                    # Check it doesn't conflict with other
                    if other in conflict[c]:
                        continue
                    miss = 0
                    for u in unique_to_replace:
                        if u not in neigh_set[c]:
                            miss += 1
                            if miss >= best_single_miss:
                                break
                    if miss < best_single_miss:
                        best_single_miss = miss
                        best_single = c
                    if miss == 0:
                        break
                
                if best_single != -1 and best_single_miss == 0:
                    code[idx] = best_single
                    code_set.discard(c_to_replace)
                    code_set.add(best_single)
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
                    # Recompute conflict pairs
                    conflict_pairs = []
                    for i in range(len(code)):
                        for j in range(i + 1, len(code)):
                            if code[j] in conflict[code[i]]:
                                conflict_pairs.append((i, j))
                    break
    
    return best_code, best_size

def _final_cleanup(code, neigh, counts_func, total):
    """Remove redundant words."""
    best_set = set(code)
    best_counts = [0] * total
    for c in code:
        for u in neigh[c]:
            best_counts[u] += 1
            
    final_code = []
    for c in code:
        removable = True
        for u in neigh[c]:
            if best_counts[u] == 1:
                removable = False
                break
        if not removable:
            final_code.append(c)
            for u in neigh[c]:
                best_counts[u] -= 1
    
    return final_code

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors and conflict graph
    neigh, neigh_set, conflict, total = _precompute_neighbors(q, n, radius)
    
    # 2. Greedy independent set initialization
    code, code_set = _greedy_independent_set_init(q, n, radius, neigh, neigh_set, conflict, total, rng)
    
    best_code = code[:]
    best_size = len(code)
    
    # 3. Clause learning local search
    code, best_size = _clause_learning_local_search(q, n, radius, neigh, neigh_set, conflict, total, code, code_set, deadline - 30, rng)
    if best_size < len(best_code):
        best_code = code[:]
    
    # 4. Conflict resolution pass
    code_set = set(code)
    code, best_size = _conflict_resolution_pass(q, n, radius, neigh, neigh_set, conflict, total, code, code_set, deadline - 10, rng)
    if best_size < len(best_code):
        best_code = code[:]
    
    # 5. Final local search on best_code
    code_set = set(best_code)
    code, best_size = _clause_learning_local_search(q, n, radius, neigh, neigh_set, conflict, total, code, code_set, deadline - 5, rng)
    if best_size < len(best_code):
        best_code = code[:]
    
    # 6. Final cleanup
    final_code = _final_cleanup(best_code, neigh, None, total)
    if len(final_code) < len(best_code):
        best_code = final_code
    
    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
