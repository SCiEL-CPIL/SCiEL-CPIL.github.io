import numpy as np
import itertools

def construct(q, n, radius):
    # The task specifies q=3, n=7, radius=1.
    # We implement a general greedy algorithm for small q, n.
    
    total_words = q ** n
    
    # Precompute neighbors for each word index.
    # An index represents a base-q number.
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius.
    # For radius=1, neighbors are the word itself and words differing in exactly one position by +-1 (mod q).
    
    # To speed up index conversion, we can use base-q representation.
    # However, for n=7, q=3, total_words=2187, we can just precompute everything once.
    
    # Function to convert index to tuple of digits
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    # Function to convert word tuple to index
    def word_to_index(word):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors for all indices
    # This might take a bit of time but is necessary for the fast update step.
    neighbors = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        word = index_to_word(i)
        # Add self
        neighbors[i].append(i)
        
        # Add neighbors at distance 1
        for pos in range(n):
            original_val = word[pos]
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (original_val + delta) % q
                new_idx = word_to_index(new_word)
                neighbors[i].append(new_idx)
                
                # Note: (original_val - delta) % q is the same set as (original_val + delta) % q for delta in 1..q-1
                # So we don't need to check negative deltas separately if we iterate 1..q-1 and use mod q.
                # Wait, for q=3, deltas are 1, 2. 
                # pos+1 and pos+2 (which is pos-1). 
                # So iterating delta in 1..q-1 covers all distinct neighbors.
                
    # State
    covered = np.zeros(total_words, dtype=bool)
    # gain[i] = number of currently uncovered words in the ball B(i, radius)
    # Initially, all words are uncovered, so gain[i] = size of ball
    ball_size = len(neighbors[0])
    gain = np.full(total_words, ball_size, dtype=np.int32)
    
    codewords = []
    
    while True:
        # Find the index with the maximum gain
        max_gain = np.max(gain)
        if max_gain == 0:
            break
        
        # Find one of the indices with max gain
        # Using argmax is fast
        k = np.argmax(gain)
        
        # Convert index k to word
        cw = index_to_word(k)
        codewords.append(cw)
        
        # Update state
        # For each word u in the ball of k that is newly covered:
        #   Mark it as covered
        #   Decrement gain for all j such that u is in B(j)
        #   Note: u is in B(j) iff j is in B(u)
        
        for u in neighbors[k]:
            if not covered[u]:
                covered[u] = True
                # u is now covered. It contributed 1 to gain[j] for all j in neighbors[u].
                # So we decrement gain[j] for all j in neighbors[u].
                for j in neighbors[u]:
                    gain[j] -= 1
                    
    return codewords
