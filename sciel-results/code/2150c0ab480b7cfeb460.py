import random
import itertools

def construct():
    def max_sidon_size(S):
        """Find the largest subset of S where all unordered pair sums (a<=b) are unique."""
        S = sorted(S)
        n = len(S)
        best = 0
        # Try all subsets from largest to smallest
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(S, size):
                sums = set()
                valid = True
                for i in range(len(subset)):
                    for j in range(i, len(subset)):
                        s = subset[i] + subset[j]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    best = size
                    found = True
                    break
            if found:
                break
        return best
    
    best_set = None
    best_score = 100
    
    # Try random sets
    random.seed(42)
    for _ in range(5000):
        S = random.sample(range(0, 129), 12)
        score = max_sidon_size(S)
        if score < best_score:
            best_score = score
            best_set = S
            if best_score <= 3:
                break
    
    if best_set is None:
        best_set = list(range(12))
    
    return best_set
