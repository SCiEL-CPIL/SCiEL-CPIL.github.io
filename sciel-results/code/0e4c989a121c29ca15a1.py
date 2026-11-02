import time
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Classic recursive construction of length 1! + 2! + ... + n!
    word = [1]
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from the current word
        # The current word contains all permutations of 1..m-1 as windows of length m-1
        # We need to iterate through them in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            # Only consider valid permutations (distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new = []
        for p in perms:
            # Construct the block: p[0] m p[1:] m p[0] ... ?
            # Standard construction: Insert m into p at every position?
            # No, the seed program does: block = list(p) + [m] + list(p)
            # This creates a sequence that contains p, and then p with m inserted?
            # Let's trace n=3 again with this logic.
            # W2 = [1, 2, 1]
            # m=3
            # Perms of W2 (len 2): (1,2), (2,1)
            # p=(1,2): block = [1, 2, 3, 1, 2] -> 12312
            # p=(2,1): block = [2, 1, 3, 2, 1] -> 21321
            # new = []
            # Add 12312. new = [1,2,3,1,2]
            # Add 21321.
            # Overlap check:
            # new suffix vs block prefix.
            # new=[1,2,3,1,2], block=[2,1,3,2,1]
            # o=4: new[-4:]=[2,3,1,2], block[:4]=[2,1,3,2] No.
            # o=3: new[-3:]=[3,1,2], block[:3]=[2,1,3] No.
            # o=2: new[-2:]=[1,2], block[:2]=[2,1] No.
            # o=1: new[-1:]=[2], block[:1]=[2] Yes.
            # overlap=1.
            # new += block[1:] = [1,3,2,1]
            # new = [1,2,3,1,2,1,3,2,1] -> 123121321.
            # Length 9.
            # This matches the optimal for n=3.
            
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = 0
            # Optimization: We only need to check up to min(len(new), len(block))
            # But we can break early if we find a large overlap?
            # The loop in seed goes from min down to 0.
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
            
        word = new
        
    return "".join(map(str, word))
