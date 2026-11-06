import time
import random

def construct():
    # Start with the 32-word seed
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Add two zero bits to make it 18 bits? No, the task is 17 bits.
    # The seed is 16 bits. The task is 17 bits.
    # The prompt says "extended by two zero bits" for the seed?
    # "First-order Reed-Muller code on 16 coordinates, extended by two zero bits."
    # 16 + 2 = 18 bits.
    # But the task is 17 bits.
    # "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate".
    # So the seed in the prompt is likely 18 bits, shortened to 17?
    # The current program returns 16-bit words.
    # "Each integer encodes one N-bit binary codeword, including leading zeros."
    # N=17.
    # The current program returns 16-bit words.
    # I should extend the seed to 17 bits by appending a 0 or 1?
    # The prompt says "extended by two zero bits". So 16+2=18.
    # If I shorten by one, I get 17.
    # So I should take the 18-bit words and remove one coordinate?
    # Or just append one 0 bit to the 16-bit words to make them 17-bit?
    # If I append a 0, the distance remains the same.
    # So I can just append a 0 to each 16-bit word.
    
    # Let's modify the seed generation to produce 17-bit words.
    words_17 = [w << 1 for w in words]  # Append a 0 bit
    
    # Now we have 32 words of length 17.
    # We want to find a larger set.
    # Greedy search.
    
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds
    
    # Precompute bit counts for speed?
    # Use a set for fast lookup of existing words.
    word_set = set(words_17)
    
    # Candidate generation: Random words with weight 8 or 9?
    # Or all words?
    # Let's try random words with weight 8 or 9.
    
    # To ensure we find a large code, we might need a more structured approach.
    # But let's try random first.
    
    while time.time() < deadline:
        # Generate a random candidate
        # Weight 8 or 9
        weight = random.choice([8, 9])
        positions = random.sample(range(17), weight)
        candidate = 0
        for p in positions:
            candidate |= (1 << p)
            
        if candidate in word_set:
            continue
            
        # Check compatibility
        compatible = True
        for w in word_set:
            if (w ^ candidate).bit_count() < 6:
                compatible = False
                break
                
        if compatible:
            word_set.add(candidate)
            words_17.append(candidate)
            
    return list(word_set)
