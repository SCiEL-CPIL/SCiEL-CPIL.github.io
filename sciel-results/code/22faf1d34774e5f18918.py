import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    deadline = time.time() + 55.0
    
    # 1. Extended QR Code Construction
    try:
        # Quadratic residues mod 17
        residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
        v = np.zeros(17, dtype=np.uint8)
        v[residues] = 1
        
        G = np.zeros((9, 18), dtype=np.uint8)
        for i in range(9):
            row = np.roll(v, i)
            G[i, :17] = row
            G[i, 17] = row.sum() % 2
        
        # Precompute int versions of rows for speed
        row_ints = []
        for j in range(9):
            val = 0
            for k in range(18):
                if G[j, k]:
                    val |= (1 << k)
            row_ints.append(val)
        
        codes = []
        for m in range(512):
            c = 0
            for j in range(9):
                if (m >> j) & 1:
                    c ^= row_ints[j]
            codes.append(c)
        
        # Verify all pairs
        min_d = N + 1
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                d = (codes[i] ^ codes[j]).bit_count()
                if d < min_d:
                    min_d = d
                if min_d < MIN_DIST:
                    break
            if min_d < MIN_DIST:
                break
        
        if min_d >= MIN_DIST:
            # Try to improve via local search
            improved = _local_search(codes, N, MIN_DIST, deadline)
            return improved
        else:
            pass
    except Exception:
        pass
    
    # Fallback: Optimized Greedy with conflict masks
    return _greedy_improved(N, MIN_DIST, deadline)

def _local_search(codes, N, MIN_DIST, deadline):
    """Try to improve the code via local search (add/remove/swap codewords)."""
    codes = list(codes)
    n = len(codes)
    
    # Build distance matrix for current code
    def min_dist_to_all(word, exclude_idx=None):
        """Find minimum distance from word to all codewords (excluding exclude_idx)."""
        min_d = N + 1
        for i, c in enumerate(codes):
            if i == exclude_idx:
                continue
            d = (word ^ c).bit_count()
            if d < min_d:
                min_d = d
                if min_d < MIN_DIST:
                    break
        return min_d
    
    # Try adding new codewords
    for attempt in range(1000):
        if time.time() > deadline:
            break
        
        # Random candidate
        candidate = random.getrandbits(N)
        
        # Check if it can be added
        ok = True
        for c in codes:
            d = (candidate ^ c).bit_count()
            if d < MIN_DIST:
                ok = False
                break
        
        if ok:
            codes.append(candidate)
    
    # Try swapping codewords to allow more additions
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Try removing a random codeword and see if we can add more
        idx = random.randrange(len(codes))
        removed = codes.pop(idx)
        
        # Try to add several new codewords
        added = 0
        for _ in range(20):
            if time.time() > deadline:
                break
            candidate = random.getrandbits(N)
            ok = True
            for c in codes:
                d = (candidate ^ c).bit_count()
                if d < MIN_DIST:
                    ok = False
                    break
            if ok:
                codes.append(candidate)
                added += 1
        
        # Only keep if we gained codewords
        if added > 0:
            pass  # Keep the improvement
        else:
            # Restore
            codes.insert(idx, removed)
    
    # Final verification
    for i in range(len(codes)):
        for j in range(i+1, len(codes)):
            d = (codes[i] ^ codes[j]).bit_count()
            if d < MIN_DIST:
                # Invalid - shouldn't happen, but be safe
                return codes[:i]
    
    return codes

def _greedy_improved(N, MIN_DIST, deadline):
    """Greedy algorithm with conflict masks and multiple restarts."""
    best_codes = []
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Weight-sorted initialization: start with all-zero word
        codes = [0]
        
        # Precompute all words by weight for systematic exploration
        # But for speed, use random candidates with conflict checking
        
        # Conflict mask: for each codeword, track which positions are "blocked"
        # Actually, simpler: just check each candidate against all existing codewords
        
        # Try to add codewords greedily
        for _ in range(2000):
            if time.time() > deadline:
                break
            
            # Random candidate
            candidate = random.getrandbits(N)
            
            # Check against all existing codewords
            ok = True
            for c in codes:
                d = (candidate ^ c).bit_count()
                if d < MIN_DIST:
                    ok = False
                    break
            
            if ok:
                codes.append(candidate)
        
        if len(codes) > len(best_codes):
            best_codes = codes[:]
        
        # Early termination if we've hit a good size
        if len(best_codes) >= 512:
            break
    
    # Final verification
    for i in range(len(best_codes)):
        for j in range(i+1, len(best_codes)):
            d = (best_codes[i] ^ best_codes[j]).bit_count()
            if d < MIN_DIST:
                return best_codes[:i]
    
    return best_codes
