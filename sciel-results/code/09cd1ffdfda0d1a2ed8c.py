"""Ternary covering code K_3(8,1) - vectorized numpy incidence matrix local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    def time_left():
        return deadline - time.time()

    # ---- Build the universe: all q^n words as integer labels ----
    total = q ** n
    N = total
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)

    # ---- Build incidence matrix: M[i,j]=1 if word j is within radius of codeword i ----
    # For radius=1, q=3, n=8: each codeword covers 1 + n*(q-1) = 1+8*2 = 17 words
    # M is N x N boolean, but we store it as uint8 for memory efficiency
    # Actually N=6561, so N*N = ~43M entries * 1 byte = ~43MB, manageable
    M = np.zeros((N, N), dtype=np.uint8)
    
    # For each word i, mark which words it covers (within Hamming distance radius)
    # For radius=1: word i covers itself and all words differing in exactly 1 position
    for i in range(N):
        w = words[i]
        M[i, i] = 1
        for pos in range(n):
            for alt in range(q):
                if alt == w[pos]:
                    continue
                # Compute index of word with position pos changed to alt
                # words are in lexicographic order: word index = w[0]*q^(n-1) + w[1]*q^(n-2) + ...
                j = i
                # We need to compute the index of the neighbor
                # Since words are stored as product(range(q), repeat=n) in lex order:
                # index = w[0]*q^(n-1) + w[1]*q^(n-2) + ... + w[n-1]
                # Changing w[pos] to alt changes index by (alt - w[pos]) * q^(n-1-pos)
                j = i + (alt - w[pos]) * (q ** (n - 1 - pos))
                M[i, j] = 1
    
    # ---- Initial solution: product of two tetracodes (9*9=81 words for n=8) ----
    # Tetracode is optimal K_3(4,1) = 9
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    # For n=8, use product structure: codeword = (tetracode_word, tetracode_word)
    # This gives 9*9 = 81 codewords, but coverage of product is NOT simply product of coverages
    # Actually for covering radius 1, the product code covers words within distance 1 of 
    # some (t1, t2) pair. A word (u, v) is covered if d(u,t1)+d(v,t2) <= 1 for some (t1,t2).
    # This means d(u,t1)<=1 AND d(v,t2)<=1, with at most one of them being exactly 1.
    # So the product of two radius-1 codes gives a radius-1 code for the product space.
    # Wait - that's not right. If d(u,t1)=1 and d(v,t2)=1, then d((u,v),(t1,t2))=2 > 1.
    # So the product does NOT work as a covering code for radius 1!
    
    # Instead, use the direct sum construction from the seed: tetracode × {0,1,2}^4
    # This gives 9 * 81 = 729... no wait, 9 * 3^4 = 9*81 = 729. That's too many.
    # The seed uses tetracode(9) × {0,1,2}^{n-4} = 9 * 3^4 = 729 for n=8.
    # But the seed says 486. Let me recheck: 9 * 3^4 = 729, but the seed comment says 486.
    # Actually looking again: the seed does for h in tetracode (9 words) for tail in product(range(3), repeat=n-4)
    # For n=8, n-4=4, so 9 * 81 = 729. But the task says best known is 486.
    # Hmm, maybe I'm miscounting. Let me just use the seed's initialization.
    
    init_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            init_codewords.append(tuple(h + list(tail)))
    
    # Convert to indices
    selected_set = set()
    for cw in init_codewords:
        # Find index: cw is in lex order
        idx = 0
        for pos in range(n):
            idx = idx * q + cw[pos]
        selected_set.add(idx)
    
    selected = np.zeros(N, dtype=bool)
    for idx in selected_set:
        selected[idx] = True
    
    # ---- Verify coverage using vectorized operations ----
    # Covered words = OR of M[i,:] for all selected i
    # Use matrix-vector multiplication: coverage = M[selected, :].any(axis=0)
    coverage = M[selected, :].any(axis=0)
    
    if not coverage.all():
        # Greedy add missing
        uncovered_count = N - coverage.sum()
        while uncovered_count > 0 and time_left() > 5.0:
            # Find best codeword to add: maximizes newly covered words
            # gain[i] = number of currently uncovered words covered by i
            # uncovered_mask = ~coverage
            uncovered_mask = ~coverage
            gains = M[:, uncovered_mask].sum(axis=1)
            # Only consider non-selected codewords
            gains[selected] = -1
            best = np.argmax(gains)
            if gains[best] <= 0:
                break
            selected[best] = True
            coverage |= M[best, :]
            uncovered_count = N - coverage.sum()
    
    # ---- Local search: vectorized greedy removal ----
    # For each selected codeword i, check if M[i,:] is subset of OR of M[j,:] for j!=i selected
    # Equivalently: is there any word covered ONLY by i?
    # count[w] = number of selected codewords covering word w
    # i is redundant iff for all w in M[i,:], count[w] >= 2
    
    improved = True
    removal_passes = 0
    while improved and time_left() > 10.0:
        improved = False
        removal_passes += 1
        # Compute coverage count for each word
        count = M[selected, :].sum(axis=0)
        sel_indices = np.where(selected)[0]
        for i in sel_indices:
            if time_left() < 10.0:
                break
            # Check if any word is covered only by i
            # Words covered by i: M[i, :] == 1
            # Among those, is any with count == 1?
            only_by_i = (M[i, :] == 1) & (count == 1)
            if not only_by_i.any():
                # i is redundant, remove it
                selected[i] = False
                coverage = M[selected, :].any(axis=0)
                improved = True
    
    # ---- 1-swap local search: remove one, try to add one to compensate ----
    swap_iterations = 0
    max_swap_iters = 50
    while time_left() > 15.0 and swap_iterations < max_swap_iters:
        swap_iterations += 1
        found_improvement = False
        sel_indices = np.where(selected)[0]
        
        for i in sel_indices:
            if time_left() < 15.0:
                break
            # Try removing i
            selected[i] = False
            new_coverage = M[selected, :].any(axis=0)
            uncovered = ~new_coverage
            uncovered_count = uncovered.sum()
            
            if uncovered_count == 0:
                # Pure removal worked!
                found_improvement = True
                break
            
            # Try to find a single codeword j that covers all uncovered words
            # j must cover all words in uncovered: M[j, uncovered].all()
            # And j should not be already selected
            candidate_mask = ~selected
            # For efficiency, first filter: j must cover at least most uncovered
            if uncovered_count <= 20:
                # Check candidates that cover all uncovered
                covers_all = M[:, uncovered].all(axis=1)
                candidates = np.where(covers_all & candidate_mask)[0]
                if len(candidates) > 0:
                    # Pick the one that covers the most total words (to be safe)
                    total_covers = M[candidates, :].sum(axis=1)
                    best_j = candidates[np.argmax(total_covers)]
                    selected[best_j] = True
                    found_improvement = True
                    break
            else:
                # Too many uncovered for single swap to fix easily
                # Restore i
                selected[i] = True
                continue
        
        if not found_improvement:
            # No improvement in this pass
            if swap_iterations > 10:
                break
    
    # ---- Final greedy removal pass ----
    improved = True
    while improved and time_left() > 5.0:
        improved = False
        count = M[selected, :].sum(axis=0)
        sel_indices = np.where(selected)[0]
        for i in sel_indices:
            if time_left() < 5.0:
                break
            only_by_i = (M[i, :] == 1) & (count == 1)
            if not only_by_i.any():
                selected[i] = False
                coverage = M[selected, :].any(axis=0)
                improved = True
    
    # ---- Final verification ----
    coverage = M[selected, :].any(axis=0)
    if not coverage.all():
        # Emergency fix
        uncovered_mask = ~coverage
        while uncovered_mask.any() and time_left() > 2.0:
            gains = M[:, uncovered_mask].sum(axis=1)
            gains[selected] = -1
            best = np.argmax(gains)
            if gains[best] <= 0:
                break
            selected[best] = True
            coverage |= M[best, :]
            uncovered_mask = ~coverage
    
    # ---- Return result ----
    sel_indices = np.where(selected)[0]
    return [words[idx].tolist() for idx in sel_indices]
# EVOLVE-BLOCK-END
