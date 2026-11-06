import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    deadline = time.time() + 55
    
    # Start with 32-word RM code
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
            
    code_array = np.zeros((len(words), N), dtype=np.uint8)
    for i, w in enumerate(words):
        for j in range(N):
            code_array[i, j] = (w >> j) & 1
            
    while time.time() < deadline:
        # Generate 5000 random words
        batch_size = 5000
        rand_bits = np.random.randint(0, 2, size=(batch_size, N))
        
        # Check distance
        # If code_array is large, this might be slow.
        # M is at most ~300.
        # 300 * 5000 * 17 = 25.5 million operations.
        # This is fast in NumPy.
        
        diff = code_array[:, np.newaxis, :] ^ rand_bits[np.newaxis, :, :]
        dists = np.sum(diff, axis=2)
        min_dists = np.min(dists, axis=0)
        
        valid_indices = np.where(min_dists >= MIN_DIST)[0]
        
        if len(valid_indices) > 0:
            new_words = rand_bits[valid_indices]
            code_array = np.vstack((code_array, new_words))
            
    final_words = []
    for i in range(code_array.shape[0]):
        w = 0
        for j in range(N):
            if code_array[i, j]:
                w |= (1 << j)
        final_words.append(w)
        
    return final_words
