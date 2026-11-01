import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    if n <= 4:
        # Standard construction is optimal for n<=4
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) 
                              if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 5, use the standard recursive construction
    # This produces length sum_{k=1}^n k! which for n=7 is 5913
    # The shortest known is 5905, so there's a gap of 8 to close
    
    def recursive_superperm(n):
        """Generate superpermutation using the standard recursive construction."""
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) 
                              if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word
    
    # Generate the standard superpermutation
    word = recursive_superperm(n)
    current_len = len(word)
    
    # For n=7, try to find local improvements
    # The standard construction has a specific structure where we can potentially
    # find better overlaps by examining the permutation sequence
    
    if n == 7 and time.time() < deadline:
        # Extract the sequence of permutations from the word
        # In a superpermutation, each permutation appears as a window of length n
        perm_seq = []
        seen_perms = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if window not in seen_perms:
                seen_perms.add(window)
                perm_seq.append(window)
        
        # Verify we have all 5040 permutations
        if len(perm_seq) == 5040:
            # Try to improve by finding better overlaps
            # The length is determined by: n + sum(n - overlap(perm_seq[i], perm_seq[i+1]))
            # We want to maximize the sum of overlaps
            
            def calc_overlap(p1, p2, n):
                """Calculate the maximum overlap between two permutations."""
                for k in range(n-1, 0, -1):
                    if p1[n-k:] == p2[:k]:
                        return k
                return 0
            
            # Current total overlap
            current_overlap = sum(calc_overlap(perm_seq[i], perm_seq[i+1], n) 
                                  for i in range(len(perm_seq)-1))
            
            # Try local swaps to improve overlap
            # Focus on adjacent permutations where overlap is low
            improved = False
            for _ in range(2000):  # Limited iterations due to time
                if time.time() > deadline - 5:
                    break
                
                # Find a random pair of adjacent permutations to try swapping
                idx = __import__('random').randint(0, len(perm_seq) - 2)
                if idx == 0 or idx == len(perm_seq) - 2:
                    continue
                
                # Calculate overlap change if we swap perm_seq[idx] and perm_seq[idx+1]
                old_overlap = (calc_overlap(perm_seq[idx-1], perm_seq[idx], n) + 
                              calc_overlap(perm_seq[idx], perm_seq[idx+1], n) + 
                              calc_overlap(perm_seq[idx+1], perm_seq[idx+2], n))
                
                new_overlap = (calc_overlap(perm_seq[idx-1], perm_seq[idx+1], n) + 
                              calc_overlap(perm_seq[idx+1], perm_seq[idx], n) + 
                              calc_overlap(perm_seq[idx], perm_seq[idx+2], n))
                
                if new_overlap > old_overlap:
                    perm_seq[idx], perm_seq[idx+1] = perm_seq[idx+1], perm_seq[idx]
                    improved = True
            
            # Reconstruct word from improved permutation sequence if improved
            if improved:
                new_word = list(perm_seq[0])
                for i in range(1, len(perm_seq)):
                    p1 = perm_seq[i-1]
                    p2 = perm_seq[i]
                    # Find maximum overlap
                    max_k = 0
                    for k in range(n-1, 0, -1):
                        if p1[n-k:] == p2[:k]:
                            max_k = k
                            break
                    new_word.extend(p2[max_k:])
                
                if len(new_word) < current_len:
                    word = new_word
                    current_len = len(new_word)
    
    return "".join(map(str, word))
