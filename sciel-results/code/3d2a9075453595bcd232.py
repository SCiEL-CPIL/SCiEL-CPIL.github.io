"""Dual greedy set cover with shadow-price weighting for ternary covering codes."""
import time
import random
import numpy as np
import heapq
from collections import defaultdict

def construct(q, n, radius):
    """
    Construct a covering code using a dual greedy set cover with shadow prices.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    deadline = time.time() + 115.0  # 115s safety margin within 120s limit
    
    # Precompute all words as integers
    total_words = q ** n
    # Each word is an integer 0..total_words-1, representing base-q representation
    
    def word_to_int(word):
        """Convert a list of digits to an integer (base q)."""
        result = 0
        for d in word:
            result = result * q + d
        return result
    
    def int_to_word(w):
        """Convert an integer to a list of base-q digits of length n."""
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = w % q
            w //= q
        return digits
    
    # Precompute neighbors: for each word, list all words within Hamming distance <= radius
    # For radius=1, neighbors are the word itself plus all words differing in exactly one position
    # For general radius, we need to generate all words within Hamming distance radius
    
    def generate_neighbors(w):
        """Generate all words within Hamming distance <= radius from word w."""
        w_int = word_to_int(int_to_word(w))
        seen = set([w_int])
        
        if radius == 0:
            return [w_int]
        
        # For radius 1: just flip one digit
        if radius == 1:
            word = int_to_word(w_int)
            for pos in range(n):
                old_val = word[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        word[pos] = new_val
                        seen.add(word_to_int(word))
                        word[pos] = old_val
            return list(seen)
        
        # For general radius: BFS in Hamming space up to distance radius
        from collections import deque
        word = int_to_word(w_int)
        queue = deque([(w_int, 0)])
        while queue:
            current, dist = queue.popleft()
            if dist >= radius:
                continue
            cur_word = int_to_word(current)
            for pos in range(n):
                old_val = cur_word[pos]
                for new_val in range(q):
                    if new_val != old_val:
                        cur_word[pos] = new_val
                        neighbor = word_to_int(cur_word)
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append((neighbor, dist + 1))
                        cur_word[pos] = old_val
        return list(seen)
    
    # Initialize coverage tracking
    # covered[word_int] = number of codewords that cover this word
    covered = np.zeros(total_words, dtype=np.int32)
    
    # candidate_count[word_int] = how many potential codewords could cover this word
    # A word x can be covered by any codeword c such that d(c, x) <= radius
    # That is, c is in the ball of radius radius around x, which has the same size as ball around c
    # So candidate_count[x] = size of ball of radius radius around x = number of words within distance radius
    
    # Precompute ball size
    ball_size = 0
    for d in range(radius + 1):
        ball_size += (q - 1) ** d
    # Actually: sum_{i=0}^{radius} C(n,i) * (q-1)^i
    ball_size = sum((lambda i: 1)(0) if i == 0 else 1 for i in range(0))  # placeholder
    from math import comb
    ball_size = 0
    for i in range(min(radius, n) + 1):
        ball_size += comb(n, i) * ((q - 1) ** i)
    
    # Initialize candidate counts: every word can potentially be covered by any word in its radius ball
    # But we need to track which specific codewords have been chosen
    # Instead, let's use a different approach:
    # For each word x, track how many NOT-YET-CHIDDEN codewords could cover it
    
    # Actually, let's use the shadow price approach:
    # price[x] = 1 / max(1, number of remaining candidates that cover x)
    # Initially, for word x, the candidates that cover it are all words in ball(x, radius)
    # So initial count = ball_size (approximately, since all those words exist in the space)
    
    # More precisely: candidates covering x = {c : d(c,x) <= radius} = ball(x, radius)
    # Size of this ball = ball_size (computed above)
    
    # We'll maintain:
    # remaining_covers[x] = number of not-yet-selected codewords that are within distance radius of x
    # Initially = ball_size for all x
    
    remaining_covers = np.full(total_words, ball_size, dtype=np.int64)
    
    # selected_codewords: set of selected codeword integers
    selected = set()
    result = []
    
    # Priority queue: (-priority, word_int) where priority = 1 / remaining_covers[word_int]
    # We want to pick words with high priority (few remaining candidates)
    
    def get_priority(x):
        return 1.0 / max(1, remaining_covers[x])
    
    # Initialize heap
    # For efficiency, we'll lazily update the heap
    heap = []
    for w in range(total_words):
        heapq.heappush(heap, (-get_priority(w), w))
    
    # To avoid processing stale entries, we'll use a version counter
    # Or we can just recompute priority when we pop from heap
    
    while time.time() < deadline:
        if len(selected) == total_words:  # All covered
            break
        
        # Get the word with highest priority
        valid = False
        while heap:
            neg_prio, w = heapq.heappop(heap)
            # Check if this word is still uncovered or has stale priority
            if covered[w] > 0:
                continue
            # Recompute priority to check if it's still valid
            current_prio = get_priority(w)
            if abs(-neg_prio - current_prio) > 1e-12:
                # Stale entry, push updated version
                heapq.heappush(heap, (-current_prio, w))
                continue
            valid = True
            break
        
        if not valid:
            break
        
        # Select codeword w
        selected.add(w)
        result.append(int_to_word(w))
        
        # Update coverage and remaining counts for all words within radius of w
        neighbors = generate_neighbors(w)
        for nb in neighbors:
            if covered[nb] == 0:
                covered[nb] = 1  # Mark as covered (we only need to know if covered or not)
            # Decrement remaining covers for nb: w is no longer a candidate
            if nb in remaining_covers and remaining_covers[nb] > 0:
                remaining_covers[nb] -= 1
                # Push updated priority for nb if it's still uncovered
                if covered[nb] == 0:
                    heapq.heappush(heap, (-get_priority(nb), nb))
        
        # Check if all words are covered
        if np.all(covered > 0):
            break
    
    # Return the list of codewords
    return result
