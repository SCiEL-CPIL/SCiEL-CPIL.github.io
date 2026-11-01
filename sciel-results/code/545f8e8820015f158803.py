"""Superpermutation construction via recursive insertion with maximal overlap."""
# EVOLVE-BLOCK-START
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        # Build via recursive construction
        word = _build_superperm(4)
        return _trim(word, 4)
    
    word = _build_superperm(n)
    return _trim(word, n)


def _build_superperm(n):
    """Build superpermutation using recursive insertion with maximal overlap."""
    word = [1]
    for m in range(2, n + 1):
        # Generate all (m-1)-permutations in lexicographic order
        perms = list(permutations(range(1, m)))
        
        # Build trie for overlap detection
        # The trie maps sequences to the last position where they appeared as a suffix
        # We process the word incrementally and track suffixes via a trie
        
        new_word = []
        # Trie: dict mapping tuple -> list of positions (we only need the last one)
        # Actually, for maximal overlap we need: given a block, find the longest prefix
        # that matches a suffix of new_word.
        # We can maintain a trie of all suffixes of new_word as we build it.
        
        # Better approach: maintain a dictionary mapping tuples to their occurrence
        # as suffixes. When we append a character, we update all suffixes.
        # For efficiency, we use a trie structure.
        
        # Let's use a different approach: maintain a set of all suffixes of new_word
        # indexed by length. For each length L, store the tuple of the last L chars.
        
        # Actually, the most efficient approach for this specific problem:
        # Since we're appending blocks one at a time, and we want the longest prefix
        # of the block that matches a suffix of new_word, we can:
        # 1. Precompute the current suffixes of new_word
        # 2. For each block, check from longest to shortest
        
        # Let me use a simpler but still efficient approach:
        # Maintain a list of all suffixes (as tuples) of the current new_word.
        # When appending a block, first check overlap, then append and update suffixes.
        
        # For n=7, the blocks are of length 2*(m-1)+1 = 13 for m=7.
        # The total word length is ~5905, so we have ~5905 suffixes to maintain.
        # Checking overlap: for each block, we check up to 13 lengths.
        # Total operations: 5040 blocks * 13 checks * O(1) tuple comparison = ~65K ops.
        
        # Let's use a dictionary: suffix_dict[length] = tuple of last `length` chars
        # But we need to check if block[:o] == suffix_dict[o] for o from min(len(block), len(new_word)) down to 0.
        
        suffix_cache = {}  # length -> tuple
        
        for p in perms:
            block = list(p) + [m] + list(p)
            block_tuple = tuple(block)
            
            # Find maximal overlap
            max_check = min(len(new_word), len(block))
            best_overlap = 0
            for o in range(max_check, 0, -1):
                # Get the last o chars of new_word
                if o in suffix_cache:
                    suffix = suffix_cache[o]
                else:
                    suffix = tuple(new_word[-o:])
                    suffix_cache[o] = suffix
                
                if suffix == block_tuple[:o]:
                    best_overlap = o
                    break
            
            # Append the non-overlapping part
            append_part = block[best_overlap:]
            new_word.extend(append_part)
            
            # Update suffix cache
            # We need to update suffixes for all lengths up to min(len(new_word), max_block_len)
            # But we can be lazy: only compute when needed.
            # Clear the cache entries that might be stale? No, the suffixes only grow.
            # Actually, suffix_cache[o] for o <= len(append_part) is now stale because
            # the last o chars have changed. For o > len(append_part), they're still valid
            # only if o <= len(append_part) is false... wait.
            
            # When we append `append_part`, the last `o` chars of new_word change for
            # all o <= len(new_word). So the entire cache is stale after each append.
            # This makes the cache useless.
            
            # Alternative: don't cache, just compute directly.
            # For n=7, len(block) = 13, and we check at most 13 overlaps per block.
            # 5040 blocks * 13 checks = 65520 tuple comparisons of length up to 13.
            # Each comparison is O(13) in the worst case. Total: ~850K operations.
            # This should be fast enough in Python.
            
            # Let's just clear the cache and not use it.
            suffix_cache.clear()
        
        word = new_word
    
    return word


def _trim(word, n):
    """Trim leading and trailing characters while maintaining validity."""
    from math import factorial
    target = factorial(n)
    
    def count_perms(w):
        """Count distinct n-permutations in word w."""
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen)
    
    # Trim from the front
    while len(word) > n and count_perms(word[1:]) == target:
        word = word[1:]
    
    # Trim from the back
    while len(word) > n and count_perms(word[:-1]) == target:
        word = word[:-1]
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
