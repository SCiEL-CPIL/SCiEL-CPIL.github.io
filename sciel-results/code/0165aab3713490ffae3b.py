<<<<<<< SEARCH
    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    # Determine number of restarts based on time budget
    start_time = time.time()
    max_time = 110.0 # Leave 10s for overhead
    
    restart_count = 0
    rng = np.random.RandomState(42)
    
    # Helper to convert idx to word
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n-1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    while True:
=======
    all_mask = (1 << total_words) - 1
    best_codewords = None
    best_len = float('inf')
    
    # Precompute all words for fast lookup
    all_words = []
    for i in range(total_words):
        word = [0] * n
        temp = i
        for k in range(n-1, -1, -1):
            word[k] = temp % q
            temp //= q
        all_words.append(word)
    
    # Determine number of restarts based on time budget
    start_time = time.time()
    max_time = 110.0 # Leave 10s for overhead
    
    restart_count = 0
    rng = np.random.RandomState(42)
    
    while True:
>>>>>>> REPLACE

