"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses ILS with structured initialization and efficient local search to minimize codewords.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    total_words = q ** n
    
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors using numpy for speed
    # For radius=1: each word has 1 + n*(q-1) neighbors
    neighbors = []
    for i in range(total_words):
        w = int_to_word(i, n, q)
        nbrs = [i]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        w[pos] = nv
                        nbrs.append(word_to_int(w, q))
                w[pos] = orig
        neighbors.append(nbrs)
    
    # Convert to numpy arrays for faster operations
    max_deg = 1 + n * (q - 1) if radius >= 1 else 1
    nbr_matrix = np.zeros((total_words, max_deg), dtype=np.int32)
    nbr_deg = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        nbr_matrix[i, :len(neighbors[i])] = neighbors[i]
        nbr_deg[i] = len(neighbors[i])
    
    all_words = list(range(total_words))
    
    # Initialize with tetracode direct sum construction
    codewords = set()
    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = word_to_int(word, q)
                    codewords.add(idx)
    else:
        # Greedy fallback
        uncovered = set(range(total_words))
        while uncovered and time.time() < deadline:
            best_c, best_cnt = -1, -1
            sample = list(uncovered)
            random.shuffle(sample)
            for cand in sample[:50]:
                cnt = sum(1 for nb in neighbors[cand] if nb in uncovered)
                if cnt > best_cnt:
                    best_cnt, best_c = cnt, cand
            if best_c < 0:
                best_c = next(iter(uncovered))
            codewords.add(best_c)
            for nb in neighbors[best_c]:
                if nb in uncovered:
                    uncovered.discard(nb)
    
    # Local search state
    def compute_cover_count(cw_set):
        cc = np.zeros(total_words, dtype=np.int32)
        for c in cw_set:
            idxs = nbr_matrix[c, :nbr_deg[c]]
            cc[idxs] += 1
        return cc
    
    def count_singles(cc):
        return int(np.sum(cc == 1))
    
    def is_removable(c, cc):
        idxs = nbr_matrix[c, :nbr_deg[c]]
        return bool(np.all(cc[idxs] > 1))
    
    def try_remove_all_redundant(cw_set, cc):
        changed = True
        while changed:
            changed = False
            for c in list(cw_set):
                if is_removable(c, cc):
                    cw_set.discard(c)
                    idxs = nbr_matrix[c, :nbr_deg[c]]
                    cc[idxs] -= 1
                    changed = True
        return cw_set, cc
    
    # ILS main loop
    best_size = float('inf')
    best_codewords = set()
    
    while time.time() < deadline:
        # Phase 1: Remove redundant
        codewords, cover_count = try_remove_all_redundant(codewords, compute_cover_count(codewords))
        
        cur_size = len(codewords)
        if cur_size < best_size:
            best_size = cur_size
            best_codewords = set(codewords)
        
        if best_size <= 73:
            break
        
        # Phase 2: Local search - try moves
        singles = count_singles(cover_count)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing first
            for c in list(codewords):
                if is_removable(c, cover_count):
                    codewords.discard(c)
                    idxs = nbr_matrix[c, :nbr_deg[c]]
                    cover_count[idxs] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try moving a codeword
            c_list = list(codewords)
            if not c_list:
                break
            c = random.choice(c_list)
            
            # Try several candidate moves
            best_move = None
            best_new_singles = singles
            
            # Sample some target positions
            num_tries = 30
            for _ in range(num_tries):
                cp = random.choice(all_words)
                if cp == c or cp in codewords:
                    continue
                
                # Simulate: remove c, add cp
                idxs_c = nbr_matrix[c, :nbr_deg[c]]
                idxs_cp = nbr_matrix[cp, :nbr_deg[cp]]
                
                # Temporarily modify
                cover_count[idxs_c] -= 1
                cover_count[idxs_cp] += 1
                
                # Check validity and count singles
                if np.any(cover_count == 0):
                    # Invalid, revert
                    cover_count[idxs_cp] -= 1
                    cover_count[idxs_c] += 1
                    continue
                
                ns = count_singles(cover_count)
                
                # Check if this enables a removal
                can_remove = False
                for cc in codewords:
                    if cc != c and is_removable(cc, cover_count):
                        can_remove = True
                        break
                
                if ns < best_new_singles or (ns == best_new_singles and can_remove):
                    best_new_singles = ns
                    best_move = cp
                
                # Revert
                cover_count[idxs_cp] -= 1
                cover_count[idxs_c] += 1
            
            if best_move is not None:
                # Apply the move
                idxs_c = nbr_matrix[c, :nbr_deg[c]]
                idxs_cp = nbr_matrix[best_move, :nbr_deg[best_move]]
                cover_count[idxs_c] -= 1
                cover_count[idxs_cp] += 1
                codewords.discard(c)
                codewords.add(best_move)
                singles = best_new_singles
                improved = True
                # After a move, try removals again
                codewords, cover_count = try_remove_all_redundant(codewords, cover_count)
                singles = count_singles(cover_count)
        
        # Phase 3: Perturbation (ILS kick)
        if time.time() > deadline:
            break
        
        # Randomly move 2-3 codewords to new positions
        k = random.randint(2, 3)
        for _ in range(k):
            if not codewords:
                break
            c = random.choice(list(codewords))
            cp = random.choice(all_words)
            if cp in codewords or cp == c:
                continue
            idxs_c = nbr_matrix[c, :nbr_deg[c]]
            idxs_cp = nbr_matrix[cp, :nbr_deg[cp]]
            cover_count[idxs_c] -= 1
            cover_count[idxs_cp] += 1
            codewords.discard(c)
            codewords.add(cp)
        
        # Fix validity after perturbation: add codewords for uncovered words
        uncovered_words = np.where(cover_count == 0)[0]
        for w in uncovered_words:
            # Find a neighbor to cover w
            # Just add w itself
            if w not in codewords:
                codewords.add(w)
                idxs = nbr_matrix[w, :nbr_deg[w]]
                cover_count[idxs] += 1
        
        # Recompute cover count to be safe
        cover_count = compute_cover_count(codewords)
    
    # Final cleanup
    best_codewords, _ = try_remove_all_redundant(best_codewords, compute_cover_count(best_codewords))
    
    result = [int_to_word(c, n, q) for c in best_codewords]
    return result
# EVOLVE-BLOCK-END
