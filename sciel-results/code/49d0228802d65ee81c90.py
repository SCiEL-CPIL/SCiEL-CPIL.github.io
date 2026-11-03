import time
import math

def _classic_superperm(n):
    """Build the classic superpermutation of length sum(k! for k=1..n)."""
    word = [1]
    for m in range(2, n + 1):
        # Collect all (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        # The window size is m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation of 1..m-1
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        # We will build the new word by appending blocks
        # Start with the first block fully?
        # Actually, the standard algorithm appends blocks to the existing word?
        # No, the standard recursive definition is:
        # S_n = S_{n-1} with each (n-1)-perm P replaced by P, n, P?
        # Or is it built from scratch?
        
        # Let's use the standard "insertion" method which is efficient.
        # The word currently contains all (m-1)-perms.
        # We want to insert m into the word to cover all m-perms.
        
        # A more efficient way to generate the standard superperm:
        # It is known that the standard superperm can be generated recursively.
        # Let's stick to the previous logic which worked for n<=6.
        
        # Re-implementing the logic from the seed but optimized.
        # The seed logic:
        # new = []
        # for p in perms:
        #   block = list(p) + [m] + list(p)
        #   find overlap with end of new
        #   new += block[overlap:]
        # word = new
        
        # This logic replaces the word entirely at each step?
        # Yes, `word = new`.
        
        # Let's trace for n=3.
        # m=2:
        # word=[1]. perms of size 1: [(1,)].
        # p=(1,). block=[1,2,1].
        # new=[] -> overlap 0 -> new=[1,2,1].
        # word=[1,2,1].
        
        # m=3:
        # word=[1,2,1]. perms of size 2:
        # i=0: (1,2) -> seen={(1,2)}, perms=[(1,2)]
        # i=1: (2,1) -> seen={(1,2),(2,1)}, perms=[(1,2),(2,1)]
        # p=(1,2). block=[1,2,3,1,2].
        # new=[] -> new=[1,2,3,1,2].
        # p=(2,1). block=[2,1,3,2,1].
        # end of new: ...3,1,2. start of block: 2,1...
        # Overlap: 2 matches 2? No, new ends in 2. block starts with 2.
        # new[-1:] = [2]. block[:1] = [2]. Match.
        # Check longer: new[-2:] = [1,2]. block[:2] = [2,1]. No.
        # So overlap=1.
        # new += block[1:] -> [1,2,3,1,2,1,3,2,1].
        # Length 9. 1!+2!+3! = 1+2+6=9. Correct.
        
        # This algorithm works.
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new_word and start of block
            overlap = 0
            max_ov = min(len(new_word), len(block) - 1)
            for o in range(max_ov, -1, -1):
                if o == 0 or new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n <= 1:
        return "1"
    
    # Generate classic superpermutation
    word = _classic_superperm(n)
    
    # For n=7, length is 5913.
    # We can try to optimize with local search if time permits.
    # But local search on 5913 length string is risky.
    # Let's just return the classic one.
    # It is guaranteed valid.
    
    return "".join(map(str, word))
