    # Shuffle all_words to randomize tie-breaking in greedy phase
    all_words_shuffled = all_words[:]
    random.shuffle(all_words_shuffled)
    
    while uncovered:
        best_word = None
        best_count = -1
        for w in all_words_shuffled:
            # ...
