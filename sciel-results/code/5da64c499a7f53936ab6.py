import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    total_words = q ** n
    # Precompute neighbors for all words
    # Word is an integer 0..total_words-1
    # Convert to base q to get digits
    # This might be slow for large n, but n=7, q=3 is small.
    
    def to_digits(num):
        digits = []
        for _ in range(n):
            digits.append(num % q)
            num //= q
        return digits

    def to_num(digits):
        num = 0
        multiplier = 1
        for d in digits:
            num += d * multiplier
            multiplier *= q
        return num

    # Generate neighbors for each word
    # neighbors[i] is a list of words within distance 'radius' of i
    # For radius 1, neighbors are i itself and i with one digit changed.
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        digits = to_digits(i)
        neighbors[i].append(i)
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    digits[pos] = new_val
                    neighbors[i].append(to_num(digits))
            digits[pos] = original
            
    # Greedy Add
    code = []
    covered = [False] * total_words
    uncovered_count = total_words
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
            
        best_word = -1
        best_coverage = -1
        
        # Find word with max uncovered neighbors
        for i in range(total_words):
            if covered[i]:
                # Optimization: if i is covered, it might still be a good center?
                # No, we only care about covering UNCOVERED words.
                # A covered word doesn't help cover other uncovered words more than an uncovered one would?
                # Actually, the set of neighbors is fixed.
                # We want to maximize sum(covered[j] == False for j in neighbors[i])
                pass
            
            count = 0
            for j in neighbors[i]:
                if not covered[j]:
                    count += 1
            if count > best_coverage:
                best_coverage = count
                best_word = i
                
        if best_word == -1:
            break
            
        code.append(best_word)
        for j in neighbors[best_word]:
            if not covered[j]:
                covered[j] = True
                uncovered_count -= 1
                
    # Greedy Remove
    # Try to remove codewords that are redundant
    # We need to know for each word, how many codewords cover it.
    # And for each codeword, which words it covers.
    
    # This simple greedy remove might be O(|Code| * |Code| * SphereSize)
    # |Code| ~ 200. SphereSize 15. 200*200*15 = 600,000. Fine.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        # Rebuild coverage counts
        cover_count = [0] * total_words
        for c in code:
            for j in neighbors[c]:
                cover_count[j] += 1
                
        new_code = []
        for c in code:
            # Check if c is redundant
            # c is redundant if all j in neighbors[c] have cover_count[j] > 1
            redundant = True
            for j in neighbors[c]:
                if cover_count[j] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_code.append(c)
            else:
                # Remove c
                changed = True
                # Decrement counts
                for j in neighbors[c]:
                    cover_count[j] -= 1
                    
        code = new_code
        
    # Convert back to list of lists
    result = [to_digits(c) for c in code]
    return result
