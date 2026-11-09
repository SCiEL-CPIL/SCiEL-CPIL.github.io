import time
import sys
from collections import deque, defaultdict

def construct(n):
    """
    Constructs a superpermutation for n symbols using the classic recursive
    construction, then attempts local shortening to reduce length.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline - 40:  # Leave time for shortening
            break
        
        # Extract all unique permutations of length m-1 from the current word
        # in order of first appearance
        perms, seen = [], set()
        k = m - 1
        full_mask = (1 << k) - 1
        
        for i in range(len(word) - k + 1):
            window = word[i:i + k]
            mask = 0
            valid = True
            for sym in window:
                bit = 1 << (sym - 1)
                if mask & bit:
                    valid = False
                    break
                mask |= bit
            if valid and mask == full_mask:
                wkey = tuple(window)
                if wkey not in seen:
                    seen.add(wkey)
                    perms.append(window)
        
        new = []
        for p in perms:
            # Construct the block: p, m, p
            block = p + [m] + p
            
            # Find maximum overlap with the current end of 'new'
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
    
    # Local shortening pass
    if time.time() < deadline - 5:
        word = _local_shorten(word, n, deadline)
    
    return "".join(map(str, word))


def _local_shorten(word, n, deadline):
    """
    Attempt to shorten the superpermutation by removing redundant symbols.
    Full rebuild after each removal for correctness.
    """
    word = list(word)
    target = _factorial(n)
    full_mask = (1 << n) - 1
    
    def _build_maps(w):
        pc = defaultdict(int)
        pp = defaultdict(set)
        for i in range(len(w) - n + 1):
            window = w[i:i + n]
            mask = 0
            ok = True
            for s in window:
                b = 1 << (s - 1)
                if mask & b:
                    ok = False
                    break
                mask |= b
            if ok and mask == full_mask:
                key = tuple(window)
                pc[key] += 1
                for pos in range(i, i + n):
                    pp[pos].add(key)
        return pc, pp
    
    def _valid_count(pc):
        return sum(1 for c in pc.values() if c > 0)
    
    pc, pp = _build_maps(word)
    
    if _valid_count(pc) < target:
        return word
    
    i = 0
    removed = 0
    while i < len(word) - 1 and time.time() < deadline - 3:
        can_remove = True
        for perm in pp.get(i, ()):
            if pc[perm] <= 1:
                can_remove = False
                break
        
        if can_remove:
            saved = word[i]
            word.pop(i)
            removed += 1
            pc, pp = _build_maps(word)
            if _valid_count(pc) < target:
                word.insert(i, saved)
                pc, pp = _build_maps(word)
                break
        else:
            i += 1
    
    return word


def _factorial(n):
    """Compute n!"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
