import itertools
import time

def construct(n):
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Local Search (Greedy Deletion)
    if n < 4: # Small n, standard is optimal
        return "".join(map(str, word))
        
    # Precompute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    
    def get_counts(w):
        counts = {p: 0 for p in all_perms}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in counts:
                counts[p] += 1
        return counts

    current_word = word
    current_counts = get_counts(current_word)
    
    # Helper to check if removal at index i is valid
    def try_remove(w, counts, i):
        # Affected windows are those containing index i
        # Window j covers indices j to j+n-1
        # i is in window j if j <= i <= j+n-1 => i-n+1 <= j <= i
        start_j = max(0, i - n + 1)
        end_j = min(len(w) - n, i)
        
        # Check if any permutation count becomes 0
        affected = []
        for j in range(start_j, end_j + 1):
            p = tuple(w[j:j+n])
            if p not in counts: continue # Should not happen for valid w
            if counts[p] == 1:
                # This permutation would be lost
                return False
            affected.append((j, p))
        
        # If valid, we need to update counts
        # But for the check, we just return True
        return True

    # Iterative deletion
    max_iterations = 100
    for _ in range(max_iterations):
        improved = False
        # Try removing from left to right
        i = 0
        while i < len(current_word):
            if try_remove(current_word, current_counts, i):
                # Perform removal
                # Update counts
                start_j = max(0, i - n + 1)
                end_j = min(len(current_word) - n, i)
                for j in range(start_j, end_j + 1):
                    p = tuple(current_word[j:j+n])
                    current_counts[p] -= 1
                
                # Remove from list
                current_word.pop(i)
                improved = True
                # Do not increment i, check same index again (which is now the next element)
            else:
                i += 1
        
        if not improved:
            # Try right to left
            i = len(current_word) - 1
            while i >= 0:
                if try_remove(current_word, current_counts, i):
                    start_j = max(0, i - n + 1)
                    end_j = min(len(current_word) - n, i)
                    for j in range(start_j, end_j + 1):
                        p = tuple(current_word[j:j+n])
                        current_counts[p] -= 1
                    
                    current_word.pop(i)
                    improved = True
                else:
                    i -= 1
            
            if not improved:
                break

    return "".join(map(str, current_word))
