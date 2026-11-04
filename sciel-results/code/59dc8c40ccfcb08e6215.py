"""Ternary covering code K_3(8,1) via conflict-driven local search on point assignments."""
import time
import itertools
import numpy as np
import random


def construct(q, n, radius):
    """
    Build a covering code of length n over alphabet size q with covering radius.
    For q=3, n=8, radius=1, use a structural approach: assign each space point to a
    codeword, then minimize the number of distinct codewords while maintaining coverage.
    """
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1()
    return _greedy_general(q, n, radius)


def _solve_ternary_8_1():
    start_time = time.time()
    deadline = start_time + 112.0

    q = 3
    n = 8
    total = q ** n  # 6561

    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))

    # word_to_idx mapping
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors for each word (itself + distance 1)
    # Each word has 1 + 8*2 = 17 neighbors
    neighbors = np.zeros((total, 17), dtype=np.int32)
    for i in range(total):
        w = all_words[i]
        nbrs = [i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    nbrs.append(word_to_idx[tuple(new_w)])
        neighbors[i] = nbrs

    # For fast membership: for each word, the set of words that can cover it (i.e., its neighbors)
    # This is the same as neighbors, since coverage is symmetric in radius 1.
    # A codeword c covers word w iff w is in neighbors[c], which iff c is in neighbors[w].

    # Strategy: 
    # 1. Greedy set cover to get an initial solution
    # 2. Convert to assignment: each point is assigned to one codeword that covers it
    # 3. Local search: try to reduce the number of distinct codewords by reassigning points

    # --- Phase 1: Greedy set cover ---
    covered = np.zeros(total, dtype=np.uint8)
    uncovered_count = np.full(total, 17, dtype=np.int32)
    selected = []

    while time.time() < deadline:
        best_idx = int(np.argmax(uncovered_count))
        if uncovered_count[best_idx] == 0:
            break
        selected.append(best_idx)
        for j in neighbors[best_idx]:
            covered[j] = 1
        # Recompute uncovered counts
        # For efficiency, only recompute for candidates that might change
        # But for correctness, recompute all
        new_uc = np.zeros(total, dtype=np.int32)
        for i in range(total):
            new_uc[i] = np.sum(covered[neighbors[i]] == 0)
        uncovered_count = new_uc
        if covered.all():
            break
        if len(selected) > 600:  # safety
            break

    if not covered.all():
        # Add more codewords to cover remaining
        while not covered.all() and time.time() < deadline:
            # Find an uncovered word
            unc_idx = int(np.argmax(1 - covered))
            if covered[unc_idx]:
                break
            # Find a word that covers unc_idx and has max uncovered
            candidates = neighbors[unc_idx]  # words that can cover unc_idx
            best_c = -1
            best_uc = -1
            for c in candidates:
                uc = np.sum(covered[neighbors[c]] == 0)
                if uc > best_uc:
                    best_uc = uc
                    best_c = c
            if best_c >= 0 and best_c not in selected:
                selected.append(best_c)
                for j in neighbors[best_c]:
                    covered[j] = 1
            else:
                break

    if not covered.all():
        # Fallback: add all remaining uncovered words' neighbors
        for i in range(total):
            if not covered[i]:
                for c in neighbors[i]:
                    if c not in selected:
                        selected.append(c)
                        for j in neighbors[c]:
                            covered[j] = 1

    selected = list(set(selected))
    code_set = set(selected)

    # --- Phase 2: Build assignment ---
    # Each point w is assigned to a codeword c in code_set such that w in neighbors[c]
    # We want to minimize |code_set|
    
    # coverage_count[w] = number of codewords in code_set that cover w
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in code_set:
        for w in neighbors[c]:
            coverage_count[w] += 1

    # assignment[w] = which codeword covers w (must be in code_set and w in neighbors[assignment[w]])
    assignment = np.full(total, -1, dtype=np.int32)
    for w in range(total):
        # Find a codeword in code_set that covers w
        for c in code_set:
            if w in neighbors[c]:
                assignment[w] = c
                break
        if assignment[w] == -1:
            # Should not happen
            assignment[w] = w

    # --- Phase 3: Local search to reduce code size ---
    # Idea: try to find a codeword c that can be removed.
    # c can be removed if for every w with assignment[w] == c, there exists another codeword c' in code_set
    # such that w in neighbors[c'].
    # More generally, we can reassign w to c' and then remove c if no other w is assigned to c.

    # Better: iteratively try to remove codewords.
    # For each codeword c, check if all its assigned points can be reassigned to other codewords.
    
    # Use a different approach: 
    # While we can improve, try to find a codeword c such that removing it and reassigning its points
    # to other codewords is valid.
    
    # To make this efficient, for each point w, precompute which codewords in code_set cover it.
    # But code_set changes, so we maintain it dynamically.

    # Let's use a simpler iterative approach:
    # 1. Try to remove each codeword. If successful, remove it and reassign its points.
    # 2. Repeat until no more removals possible.
    # 3. Then try "swap" moves: replace one codeword with another to enable further removals.

    def get_coverers(w, current_code_set):
        """Return list of codewords in current_code_set that cover w."""
        # w is covered by c iff c is in neighbors[w]
        result = []
        for c in neighbors[w]:
            if c in current_code_set:
                result.append(c)
        return result

    # For efficiency, maintain for each w the list of codewords that cover it.
    # But this is expensive to update. Instead, we'll use the neighbors array.

    # Phase 3a: Remove redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in list(code_set):
            if time.time() > deadline:
                break
            # Check if c is redundant: for all w with assignment[w] == c, there's another coverer
            # Actually, we should check if c is needed: is there any w such that c is the ONLY coverer?
            # If yes, c cannot be removed.
            # If no, c can be removed, and we reassign its points.
            
            # Find points assigned to c
            points_of_c = np.where(assignment == c)[0]
            if len(points_of_c) == 0:
                # c covers no points in the assignment, so it's redundant
                code_set.remove(c)
                # Update coverage_count
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                improved = True
                break
            
            # Check if any point of c has c as its only coverer
            can_remove = True
            for w in points_of_c:
                coverers = get_coverers(w, code_set)
                if len(coverers) == 1 and coverers[0] == c:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c and reassign its points
                for w in points_of_c:
                    coverers = get_coverers(w, code_set - {c})
                    if coverers:
                        assignment[w] = coverers[0]
                    # else: shouldn't happen since can_remove was True
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                code_set.remove(c)
                improved = True
                break

    # Phase 3b: Try to reduce by finding better assignments
    # This is where the structural change comes in: we treat the assignment as the primary object.
    # We can try to "merge" two codewords by reassigning points from one to the other.
    
    # For each pair of codewords (c1, c2), check if we can reassign all points of c1 to c2
    # (and possibly other codewords) such that c1 becomes empty.
    
    # But this is O(|code_set|^2 * |points|) which is expensive.
    
    # Alternative: use a greedy approach to reassign points.
    # For each codeword c, try to move its points to other codewords.
    # If all points of c can be moved, remove c.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in list(code_set):
            if time.time() > deadline:
                break
            points_of_c = np.where(assignment == c)[0]
            if len(points_of_c) == 0:
                code_set.remove(c)
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                improved = True
                break
            
            # Try to reassign each point of c to another codeword
            all_reassigned = True
            new_assignments = {}
            for w in points_of_c:
                # Find an alternative coverer
                found = False
                for c2 in code_set:
                    if c2 == c:
                        continue
                    if w in neighbors[c2]:
                        new_assignments[w] = c2
                        found = True
                        break
                if not found:
                    all_reassigned = False
                    break
            
            if all_reassigned:
                # Perform the reassignment
                for w, c2 in new_assignments.items():
                    assignment[w] = c2
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                code_set.remove(c)
                improved = True
                break

    # Phase 3c: Try adding a new codeword to enable removal of two existing ones
    # This is more complex. For now, let's try a simpler swap:
    # Replace codeword c with a new codeword c' that covers more points, enabling removal of c.
    
    # Actually, let's try a different approach: use simulated annealing on the assignment.
    # But that might be too slow.
    
    # Let's just do a few more passes of removal.
    for _ in range(5):
        if time.time() > deadline:
            break
        improved = False
        for c in list(code_set):
            if time.time() > deadline:
                break
            points_of_c = np.where(assignment == c)[0]
            if len(points_of_c) == 0:
                code_set.remove(c)
                for w in neighbors[c]:
                    coverage_count[w] -= 1
                improved = True
                continue
            # Quick check: if c has no exclusive points, try to remove
            exclusive = 0
            for w in points_of_c:
                coverers = get_coverers(w, code_set)
                if len(coverers) == 1 and coverers[0] == c:
                    exclusive += 1
                    break
            if exclusive == 0:
                # Try to reassign
                all_reassigned = True
                for w in points_of_c:
                    found = False
                    for c2 in code_set:
                        if c2 == c and w in neighbors[c2]:
                            continue
                        if c2 != c and w in neighbors[c2]:
                            found = True
                            break
                    if not found:
                        all_reassigned = False
                        break
                if all_reassigned:
                    for w in points_of_c:
                        for c2 in code_set:
                            if c2 != c and w in neighbors[c2]:
                                assignment[w] = c2
                                break
                    for w in neighbors[c]:
                        coverage_count[w] -= 1
                    code_set.remove(c)
                    improved = True
                    break

    # Extract codewords
    codewords = [list(all_words[c]) for c in sorted(code_set)]
    return codewords


def _greedy_general(q, n, radius):
    """General greedy set cover for other parameters."""
    start_time = time.time()
    deadline = start_time + 110.0
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
    
    def get_neighbors(w):
        yield tuple(w)
        if radius >= 1:
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        new = list(w)
                        new[i] = v
                        yield tuple(new)
    
    covered = [False] * total
    codewords = []
    neighbors_list = []
    for i in range(total):
        nbrs = set()
        for cand in get_neighbors(all_words[i]):
            nbrs.add(word_to_idx(cand))
        neighbors_list.append(nbrs)
    
    uncovered_count = [len(neighbors_list[i]) for i in range(total)]
    
    while time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i in range(total):
            if uncovered_count[i] > best_count:
                best_count = uncovered_count[i]
                best_idx = i
            if best_count == 0:
                break
        if best_idx == -1 or best_count == 0:
            break
        codewords.append(list(all_words[best_idx]))
        for j in neighbors_list[best_idx]:
            if not covered[j]:
                covered[j] = True
        new_uncovered_count = [0] * total
        for i in range(total):
            cnt = 0
            for j in neighbors_list[i]:
                if not covered[j]:
                    cnt += 1
            new_uncovered_count[i] = cnt
        uncovered_count = new_uncovered_count
        if all(covered):
            break
    return codewords
