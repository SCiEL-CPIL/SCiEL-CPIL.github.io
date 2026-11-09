import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    if n <= 1:
        return "".join(map(str, range(1, n+1)))
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"  # length 9, optimal is 9? 1!+2!+3!=1+2+6=9
    if n == 4:
        # Classic gives 1+2+6+24=33, but better exists
        pass
    
    # For n up to 7, use greedy + local search
    all_perms = list(permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Build adjacency: for each permutation, which permutations can follow it?
    # A can be followed by B if A[1:] == B[:n-1]
    # This means B is determined by A[n-1] and the new first element
    # Actually: if A = (a1, a2, ..., an), then B must satisfy B[1:] = A[1:] = (a2,...,an)
    # So B = (x, a2, a3, ..., an) where x is the new first element
    # But B must be a permutation, so x must not be in {a2,...,an}, i.e., x = a1
    # Wait, that means B = (a1, a2, ..., an) = A. That's not right.
    
    # Let me reconsider. Window i is positions [i, i+n-1]. Window i+1 is [i+1, i+n].
    # They overlap in positions [i+1, i+n-1], which is n-1 symbols.
    # So window_i[1:] == window_{i+1}[:n-1]
    # If window_i = (p1, p2, ..., pn), then window_{i+1} = (x, p2, p3, ..., pn) for some x
    # For window_{i+1} to be a permutation of 1..n, x must be the missing element, which is p1
    # So window_{i+1} = (p1, p2, ..., pn) = window_i. That means consecutive windows are identical!
    
    # That can't be right. Let me re-examine.
    
    # Oh wait, I think I'm confusing myself. The superpermutation contains each permutation as a substring,
    # but consecutive substrings of length n in the word overlap by n-1 characters.
    # If the word is w[0], w[1], ..., w[L-1], then the windows are:
    # W_0 = (w[0], w[1], ..., w[n-1])
    # W_1 = (w[1], w[2], ..., w[n])
    # W_2 = (w[2], w[3], ..., w[n+1])
    # etc.
    # For W_i and W_{i+1} to both be permutations, we need:
    # W_i = (a1, a2, ..., an) is a permutation
    # W_{i+1} = (a2, a3, ..., an, b) is a permutation for some b
    # Since W_i is a permutation, {a1,...,an} = {1,...,n}
    # For W_{i+1} to be a permutation, {a2,...,an,b} = {1,...,n}, so b = a1
    # Thus W_{i+1} = (a2, a3, ..., an, a1), which is a cyclic shift of W_i!
    
    # So consecutive windows are cyclic shifts! This means the word is determined by its first n symbols,
    # and each subsequent symbol is the first symbol of the previous window.
    # The sequence of windows is: P, rotate_left(P), rotate_left^2(P), ...
    # After n steps, we're back to P.
    
    # So the word is periodic with period n? That means we can only get n distinct permutations
    # (the cyclic shifts), not n! permutations.
    
    # I must be misunderstanding the problem. Let me re-read.
    
    # "contains every one of the 5040 permutations of 1..7 as seven consecutive symbols"
    # "every window is checked" - I think this means every window of length n must be a valid permutation
    # (i.e., a permutation of 1..n, with no repeats).
    
    # But as I showed, if two consecutive windows of length n are both permutations of 1..n,
    # they must be cyclic shifts of each other. So the only way to have all n! permutations appear
    # is if... wait, that's impossible for n >= 3 since n! > n.
    
    # I think I'm wrong about the constraint. Let me re-read the task more carefully.
    
    # "The shortest word over the symbols 1 to 7 that contains every one of the 5040 permutations of 1..7 as seven consecutive symbols."
    
    # I think "as seven consecutive symbols" means the permutation appears as a substring of length 7,
    # but not necessarily that every length-7 substring is a permutation. Some substrings might not be
    # permutations (they might have repeated symbols).
    
    # But then "every window is checked" for validity - I think this means the evaluator checks that
    # every permutation appears as a substring, not that every substring is a permutation.
    
    # Let me reconsider the classic construction to understand the structure.
    
    # Classic construction for n=3:
    # Start with [1]
    # m=2: windows of size 1: [1]. perms = [(1,)]. new = [1,2,1]. word = [1,2,1]
    # m=3: windows of size 2 in [1,2,1]: (1,2), (2,1). Both have distinct elements.
    #   perms = [(1,2), (2,1)]
    #   For (1,2): block = [1,2,3,1,2]. new starts empty, so new = [1,2,3,1,2]
    #   For (2,1): block = [2,1,3,2,1]. Overlap with [1,2,3,1,2]: check suffix/prefix matches.
    #     new = [1,2,3,1,2], block = [2,1,3,2,1]
    #     o=5: new[-5:]=[1,2,3,1,2] vs block[:5]=[2,1,3,2,1] - no
    #     o=4: new[-4:]=[2,3,1,2] vs block[:4]=[2,1,3,2] - no
    #     o=3: new[-3:]=[3,1,2] vs block[:3]=[2,1,3] - no
    #     o=2: new[-2:]=[1,2] vs block[:2]=[2,1] - no
    #     o=1: new[-1:]=[2] vs block[:1]=[2] - yes!
    #     new += block[1:] = [1,3,2,1]. new = [1,2,3,1,2,1,3,2,1]
    # word = [1,2,3,1,2,1,3,2,1], length 9.
    
    # Check: does it contain all 6 permutations of 1,2,3?
    # Windows of size 3:
    # [1,2,3] - perm
    # [2,3,1] - perm
    # [3,1,2] - perm
    # [1,2,1] - not a perm (1 repeated)
    # [2,1,3] - perm
    # [1,3,2] - perm
    # [3,2,1] - perm
    # So 6 permutations appear, and one window [1,2,1] is not a permutation.
    
    # So the constraint is: every permutation appears as a substring, but not every substring needs to be a permutation.
    
    # This changes everything! The problem is: find the shortest string over {1,...,n} that contains
    # each of the n! permutations as a contiguous substring.
    
    # This is the classical superpermutation problem. The length is not simply related to a path in a graph
    # in the way I was thinking.
    
    # For the classic construction, the length is 1! + 2! + ... + n! = sum_{k=1}^{n} k!
    # For n=7: 1+2+6+24+120+720+5040 = 5913. Wait, let me recalculate.
    # 1! = 1, 2! = 2, 3! = 6, 4! = 24, 5! = 120, 6! = 720, 7! = 5040
    # Sum = 1+2+6+24+120+720+5040 = 5913
    
    # But the task says the classic construction gives n*n! = 7*5040 = 35280? No, the task says
    # "at most n * n! symbols" as an upper bound, and the classic construction is better.
    
    # Actually, I recall that the classic construction has length sum_{k=1}^{n} k! - 1 + 1 = sum_{k=1}^{n} k!
    # For n=3: 1+2+6=9. Yes, matches.
    # For n=7: 5913. But the best known is 5905, so we're close.
    
    # The improvement comes from finding better overlaps in the construction. The classic construction
    # can be improved by reordering the permutations at each level.
    
    # Let me implement the classic construction first, then try to improve it with local search.
    
    def classic_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Get the classic construction
    word = classic_construct(n)
    best_word = word[:]
    best_len = len(word)
    
    # Try to improve with local search: at each level m, try different orderings of perms
    # to find better overlaps
    
    if n <= 7 and time.time() < deadline:
        # For each level, try to find a better ordering
        # This is complex, so let's try a simpler approach: 
        # Use the classic construction but with randomized ordering at each level
        
        random.seed(42)
        num_trials = 0
        
        while time.time() < deadline:
            num_trials += 1
            # Try a randomized version of the construction
            word = [1]
            valid = True
            for m in range(2, n + 1):
                perms, seen = [], set()
                for i in range(len(word) - m + 2):
                    window = tuple(word[i:i + m - 1])
                    if len(set(window)) == m - 1 and window not in seen:
                        seen.add(window)
                        perms.append(window)
                
                # Randomly shuffle perms
                random.shuffle(perms)
                
                new = []
                for p in perms:
                    block = list(p) + [m] + list(p)
                    overlap = 0
                    max_ov = min(len(new), len(block))
                    for o in range(max_ov, -1, -1):
                        if o == 0 or new[len(new) - o:] == block[:o]:
                            overlap = o
                            break
                    new += block[overlap:]
                word = new
            
            # Verify that all permutations appear
            if len(word) < best_len:
                # Check validity
                nset = set(range(1, n+1))
                found = set()
                for i in range(len(word) - n + 1):
                    window = tuple(word[i:i+n])
                    if len(set(window)) == n:
                        found.add(window)
                if len(found) == len(list(permutations(range(1, n+1)))):
                    best_word = word[:]
                    best_len = len(word)
            
            # Limit trials
            if num_trials > 1000:
                break
    
    return "".join(map(str, best_word))
