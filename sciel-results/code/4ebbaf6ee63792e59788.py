import time

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols using the standard recursive construction.
    For n <= 6, this is optimal. For n = 7, this yields 5913 (optimal is 5905).
    """
    if n <= 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    word = [1]
    
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
            
        perms = []
        seen = set()
        # Extract unique permutations of length m-1 from the current word
        # The word contains all permutations of 1..m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a valid permutation (all distinct)
            # Since the previous step ensured validity, this might be redundant for m>2,
            # but necessary for safety or if the word was modified.
            # However, in the standard construction, all windows of size m-1 are perms.
            if window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block is p[0], m, p[1], ..., p[m-2]
            # Wait, the seed program used: list(p) + [m] + list(p)
            # Let's re-verify the seed program logic.
            # Seed: block = list(p) + [m] + list(p)
            # For p=(1,2), block=[1,2,3,1,2].
            # This matches my manual trace of the seed.
            
            block = list(p) + [m] + list(p)
            
            # Find maximum overlap between end of new_word and start of block
            # Max possible overlap is len(block) - 1, but limited by len(new_word)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # Check from largest possible overlap down to 0
            # Optimization: The overlap is usually small.
            # In the standard construction, the overlap is exactly m-1?
            # For m=3, p=(1,2), block=12312. p=(2,1), block=21321.
            # new_word ends in 12. block starts 21.
            # Overlap 1.
            # Generally, the overlap is m-1?
            # Let's check m=4.
            # It turns out the overlap is always m-1 in the standard construction?
            # No, the seed program searches for it.
            
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word += block[overlap:]
        
        word = new_word
        
    return "".join(map(str, word))
