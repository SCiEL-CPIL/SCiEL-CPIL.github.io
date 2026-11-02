import time
import sys

def construct(n):
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # Convert to list of permutations (windows)
    # The word is a list of integers.
    # We need the sequence of permutations P_i = word[i:i+n]
    # But we can work on the word directly?
    # Shortcut condition: 
    # If we remove word[i], the new word is word[:i] + word[i+1:]
    # The permutations covered are the windows of the new word.
    # This is complex to check directly on the word.
    # Better to work on the sequence of permutations.
    
    # Let's extract the sequence of permutations from the classic word.
    # P_0 = word[0:n], P_1 = word[1:n+1], etc.
    # The transition P_i -> P_{i+1} is always valid in the constructed word.
    
    seq = [tuple(word[i:i+n]) for i in range(len(word) - n + 1)]
    
    # Local Search: Shortcut
    # We want to remove P_i if P_{i-1}[1:] == P_{i+1}[:-1]
    # This implies P_{i-1} and P_{i+1} are adjacent in the graph.
    # If we remove P_i, we connect P_{i-1} to P_{i+1}.
    # We must ensure that P_i is not the ONLY occurrence of that permutation in the sequence.
    # If P_i appears elsewhere, we can remove it here.
    
    # Count occurrences
    from collections import Counter
    counts = Counter(seq)
    
    changed = True
    while changed:
        changed = False
        new_seq = []
        # We need to be careful with indices if we remove elements.
        # It's safer to build a new list in one pass.
        # But the condition depends on neighbors.
        # If we remove i, it affects i-1 and i+1.
        # A simple forward pass might miss opportunities or be inconsistent.
        # Let's try a simple forward pass:
        # If we can remove i, we skip it.
        # This is greedy.
        
        # Actually, let's just do a pass and if we find a removable one, remove it and restart?
        # That might be slow if many removals.
        # But N is ~6000.
        
        # Let's try a single forward pass that builds the new list.
        # If we decide to remove i, we don't add it.
        # The condition for removing i is:
        # 1. i > 0 and i < len(seq) - 1
        # 2. seq[i-1][1:] == seq[i+1][:-1]
        # 3. counts[seq[i]] > 1 (so we don't lose this permutation entirely)
        
        # However, if we remove i, the "counts" for the new sequence change.
        # But we are just checking if it appears *elsewhere* in the *original* sequence?
        # No, if we remove multiple, we might remove all copies.
        # To be safe, we should only remove if there is another copy in the *current* sequence.
        # This is tricky in a single pass.
        
        # Alternative: Just check if the permutation appears at least twice in the original seq.
        # If we remove one, there is still at least one left (unless we remove all, which we won't if we check > 1).
        # Wait, if we have [A, B, A], and we remove the first A?
        # Neighbors: None, B. Can't remove first A.
        # Remove middle B? Neighbors A, A. A[1:] == A[:-1]? Yes.
        # Is B unique? No, if B appears elsewhere.
        
        # Let's stick to the condition:
        # Remove P_i if P_{i-1} -> P_{i+1} is a valid edge AND P_i is not the last copy of its value in the sequence.
        
        # To handle the "last copy" correctly, we can pre-compute the next occurrence index for each position.
        
        # However, for simplicity and safety within time limits, let's just do a few passes of "remove if possible and not unique in original".
        # This might leave some removable ones, but it's a heuristic.
        
        # Actually, a better heuristic:
        # Iterate through the sequence.
        # If we can remove P_i, remove it.
        # Since N is small, we can afford to re-check.
        
        # Let's implement a simple loop that scans and removes.
        # To avoid index shifting issues, we can use a list of booleans "to_remove".
        
        to_remove = [False] * len(seq)
        
        # We need to know if seq[i] appears elsewhere.
        # If we mark i for removal, we should decrement the "available" count for that perm.
        # If we mark multiple, we might over-remove.
        
        # Let's do a single pass, and only remove if count > 1 in the ORIGINAL seq.
        # This is safe because if we remove all but one, that's fine.
        # If we remove the last one, that's bad.
        # So: Remove P_i only if count[seq[i]] > 1.
        # This guarantees we never remove the last copy.
        
        for i in range(1, len(seq) - 1):
            if to_remove[i]:
                continue
            # Check if we can shortcut
            if seq[i-1][1:] == seq[i+1][:-1]:
                if counts[seq[i]] > 1:
                    to_remove[i] = True
                    # We don't update counts immediately because it's a heuristic.
                    # But if we remove many, we might remove too many.
                    # To be safer, we can decrement a temporary count.
        
        # Let's refine: use a temporary count dictionary.
        temp_counts = counts.copy()
        to_remove = [False] * len(seq)
        
        for i in range(1, len(seq) - 1):
            if to_remove[i]:
                continue
            if seq[i-1][1:] == seq[i+1][:-1]:
                if temp_counts[seq[i]] > 1:
                    to_remove[i] = True
                    temp_counts[seq[i]] -= 1
        
        if any(to_remove):
            changed = True
            new_seq = [p for i, p in enumerate(seq) if not to_remove[i]]
            seq = new_seq
            # Update counts for next iteration
            counts = Counter(seq)
        else:
            break
            
    # Reconstruct word from seq
    if not seq:
        return "".join(map(str, word)) # Fallback
        
    final_word = list(seq[0])
    for i in range(1, len(seq)):
        final_word.append(seq[i][-1])
        
    return "".join(map(str, final_word))
