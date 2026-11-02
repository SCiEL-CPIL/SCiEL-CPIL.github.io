import time
import random
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe (evaluator kills at 150s)
    
    if n <= 1:
        return "1"
    
    # Generate all permutations of 1..n
    all_perms = list(permutations(range(1, n + 1)))
    N = len(all_perms)
    
    # For n=7, N=5040. We need to find a path visiting all nodes maximizing total overlap.
    # This is an asymmetric TSP (Maximizing sum of overlaps).
    
    # Precompute overlap matrix?
    # Overlap(a, b) = length of longest suffix of a that is prefix of b.
    # Since a and b are permutations of 1..n, they are distinct.
    # Max overlap is n-1.
    # Storing 5040x5040 matrix of ints takes ~ 5040^2 * 2 bytes (if short) ~ 50MB.
    # Calculating it takes 5040^2 * n ~ 25M * 7 ~ 175M ops. Might be slow in Python.
    
    # Let's try a heuristic that doesn't require the full matrix.
    # Greedy construction with look-ahead?
    # Or use the structure of permutations.
    
    # Insight: 
    # The standard recursive construction (Graham-Lagarias) is known to be optimal for n<=4 and near optimal for n=5,6.
    # For n=7, it produces length 5905.
    # The task asks for the shortest word.
    # If we can't beat 5905, we should return the standard construction.
    # Can we improve it?
    # The directive suggests "fixed-symbol anchor" decomposition.
    
    # Let's implement the standard Graham-Lagarias construction first as a baseline.
    # S_1 = "1"
    # S_{k+1} is formed by taking S_k, and for each permutation p in S_k (in order of appearance),
    # we insert p + [k+1] + p? No, that's not right.
    
    # Correct Recursive Construction:
    # Let P_k be the set of permutations of 1..k.
    # S_k is a superpermutation.
    # To get S_{k+1}:
    # For each permutation p of 1..k:
    #   Add p[0] + S_k with k+1 inserted? No.
    
    # Standard algorithm:
    # S_1 = "1"
    # S_k = concat over all permutations p of 1..k-1 of:
    #   p[0] + (S_{k-1} with k-1 removed and k inserted at position k-1?) 
    #   Actually, it's easier to generate the string directly.
    
    # Let's use the known formula for length:
    # L(n) = n! + (n-1)! + (n-2)! + ... + 1!
    # For n=7: 5040 + 720 + 120 + 24 + 6 + 2 + 1 = 5913?
    # Wait, the prompt says "shortest known word has 5905 symbols".
    # L(7) via Graham-Lagarias is 5905.
    # Let's verify:
    # 7! = 5040
    # 6! = 720
    # 5! = 120
    # 4! = 24
    # 3! = 6
    # 2! = 2
    # 1! = 1
    # Sum = 5040 + 720 + 120 + 24 + 6 + 2 + 1 = 5913.
    # Why does the prompt say 5905?
    # Ah, the recursive construction overlaps more.
    # S_{k+1} length = k * (L(k) + k + 1) + 1? No.
    # L(k+1) = L(k) + k! * k + 1?
    # L(1) = 1
    # L(2) = 3 (121, 1212? No. 121, 212? "12121" length 5? No. "121" contains 12, 21. Length 3. "12121" contains 121, 212, 121, 212.
    # Permutations of 1,2: 12, 21.
    # "121" contains 12 (pos 0), 21 (pos 1). Length 3.
    # Formula: L(n) = n! + (n-1)! + ... + 1! is for the *non-overlapping* blocks?
    # Actually, for n=4, L(4) = 33.
    # 4! + 3! + 2! + 1! = 24 + 6 + 2 + 1 = 33.
    # For n=5, L(5) = 153.
    # 120 + 24 + 6 + 2 + 1 = 153.
    # For n=6, L(6) = 873.
    # 720 + 120 + 24 + 6 + 2 + 1 = 873.
    # For n=7, L(7) = 5913.
    # But prompt says 5905.
    # "The shortest known word has 5905 symbols (A. Teodorescu, August 2026)"
    # This implies the recursive construction is NOT optimal for n=7, or my sum is wrong.
    # Wait, 5913 - 5905 = 8.
    # Maybe the recursive construction is indeed 5905?
    # Let's check the recurrence.
    # L(n) = n * L(n-1) + n! + n?
    # L(1) = 1
    # L(2) = 2*1 + 2 + 2 = 6? No, L(2)=3.
    
    # Let's just generate the standard recursive superpermutation.
    # It is guaranteed to be valid.
    # Then we can try to optimize it if time permits.
    
    def build_superperm(k):
        if k == 1:
            return [1]
        
        # Get superperm for k-1
        prev_word = build_superperm(k-1)
        prev_perms = []
        seen = set()
        for i in range(len(prev_word) - (k-1) + 1):
            window = tuple(prev_word[i:i+k-1])
            if window not in seen:
                seen.add(window)
                prev_perms.append(window)
        
        new_word = []
        for p in prev_perms:
            # Insert k into p
            # The standard construction inserts k at the end of the block?
            # Actually, it appends p[0] + (p with k appended) + ...?
            
            # Let's use the explicit construction:
            # For each permutation p of 1..k-1:
            #   Add p[0]
            #   Add the superpermutation of 1..k-1, but with k-1 replaced by k? No.
            
            # Correct construction:
            # S_k is formed by concatenating blocks B_p for each permutation p of 1..k-1.
            # B_p = p[0] + S_{k-1} with k-1 replaced by k? No.
            
            # Let's look up the algorithm.
            # "The shortest superpermutation for n symbols can be constructed recursively."
            # S_1 = "1"
            # S_{n+1} is obtained from S_n by:
            #   For each permutation p of 1..n:
            #     Add p[0]
            #     Add the string obtained by inserting n+1 at the end of p, and then the rest of S_n?
            
            # Actually, a simpler way to generate it:
            # Start with S_1 = "1"
            # For k from 2 to n:
            #   new_S = []
            #   for each permutation p of 1..k-1 (in the order they appear in S_{k-1}):
            #       new_S.append(p[0])
            #       new_S.extend(p)
            #       new_S.append(k)
            #       # Wait, this doesn't seem right.
            
            # Let's try a different approach:
            # We know the length is roughly n! + (n-1)! + ...
            # We can just use the greedy TSP approach on the full set of permutations.
            # For n=7, N=5040.
            # Greedy TSP is O(N^2) per step? No, O(N^2) total if we maintain a list of unvisited.
            # But we need to compute overlap on the fly.
            # Overlap(a, b) is O(n).
            # So Greedy is O(N^2 * n) = 5040^2 * 7 ~ 175M operations.
            # In Python, this might take ~10-20 seconds.
            # Then we can try to improve with local search.
            
            pass

    # Let's implement the Greedy TSP directly on permutations.
    # This avoids the recursive complexity and might find a better order.
    
    # State: current last permutation, set of visited permutations.
    # We want to maximize sum of overlaps.
    
    # To speed up overlap calculation:
    # Precompute for each permutation p, its suffixes of length 1..n-1 mapped to the permutation they prefix?
    # Or just compute on the fly.
    
    # Let's try the recursive construction first, as it's fast and known to be good.
    
    def standard_superperm(n):
        if n == 1:
            return [1]
        
        # Recursively get S_{n-1}
        prev_word = standard_superperm(n-1)
        
        # Extract permutations in order
        prev_perms = []
        seen = set()
        for i in range(len(prev_word) - (n-1) + 1):
            window = tuple(prev_word[i:i+n-1])
            if window not in seen:
                seen.add(window)
                prev_perms.append(window)
        
        new_word = []
        for p in prev_perms:
            # The block for p is: p[0] + (p with n appended) + (rest of p?)
            # Actually, the standard construction is:
            # For each p in prev_perms:
            #   Append p[0]
            #   Append the string formed by inserting n at the end of p, and then the rest of the previous word?
            
            # Let's use the known string for n=4:
            # 123413241234213412342143214312431243142314321432
            # Length 33.
            
            # Let's try to generate it.
            # S_1 = [1]
            # S_2:
            #   p=1. Block: 1, 12. Word: 112? No.
            #   Correct S_2 is "121".
            #   p=1. Insert 2 at end? 12.
            #   How do we get 21?
            
            # Algorithm from Wikipedia:
            # S_1 = "1"
            # S_{k+1} is formed by taking S_k and for each permutation p of 1..k:
            #   Append p[0]
            #   Append S_k with p[k-1] replaced by k+1? No.
            
            # Let's just use the Greedy TSP. It's more robust for optimization.
            
            pass

    # Greedy TSP Implementation
    # N = n!
    # For n=7, N=5040.
    # We need to be fast.
    
    # Precompute overlap?
    # Overlap(a, b) = max l such that a[-l:] == b[:l].
    # Since a, b are permutations, a[-l:] is a suffix, b[:l] is a prefix.
    # We can precompute for each permutation b, a dictionary mapping its prefix of length l to b.
    # But there are N permutations, and n prefixes each.
    # Total entries: N * n.
    # For n=7, 5040 * 7 = 35280 entries.
    # We can store: prefix_dict[l][tuple] = list of permutations starting with that prefix.
    # But we need the specific permutation b.
    
    # Alternative:
    # For each permutation a, compute its suffixes.
    # For each suffix s of a, find if any unvisited permutation b starts with s.
    # If yes, we can jump to b with overlap len(s).
    
    # This is like a trie search.
    
    # Let's build a Trie of all permutations.
    # Node: children dict, end_perm (if leaf).
    # Actually, we just need to find the best next permutation.
    
    # Let's stick to the simple O(N^2) greedy if N is small enough, or optimize.
    # For N=5040, O(N^2) is 25M checks.
    # Each check is O(n) = 7.
    # Total 175M ops.
    # Python does ~10M simple ops/sec. So ~17 seconds.
    # This is feasible within 145s.
    
    # We can improve with local search (2-opt) afterwards.
    
    # Step 1: Generate all permutations
    all_perms = list(permutations(range(1, n + 1)))
    N = len(all_perms)
    
    # Step 2: Greedy construction
    visited = [False] * N
    order = []
    
    # Start with a random permutation? Or 0?
    # Let's start with 0.
    curr = 0
    visited[0] = True
    order.append(0)
    
    # Precompute suffixes of current permutation?
    # For each step, we need to find unvisited j that maximizes overlap(curr, j).
    
    # To speed up, we can group permutations by their prefix.
    # But let's just do the naive loop first and see if it's fast enough.
    # If not, we'll optimize.
    
    # Optimization:
    # Instead of checking all unvisited j, we can check only those that share a prefix with the suffix of curr.
    # Suffixes of curr: curr[-1:], curr[-2:], ...
    # For a given suffix s, find all permutations starting with s.
    
    # Let's build a dictionary: prefix -> list of indices
    # prefix is a tuple.
    prefix_map = {}
    for i, p in enumerate(all_perms):
        for l in range(1, n): # Length 1 to n-1
            pref = p[:l]
            if pref not in prefix_map:
                prefix_map[pref] = []
            prefix_map[pref].append(i)
    
    # Now, for current permutation curr_idx:
    # curr = all_perms[curr_idx]
    # Check suffixes of curr from longest to shortest.
    # For each suffix s = curr[-l:]:
    #   If s is in prefix_map:
    #     For each j in prefix_map[s]:
    #       If not visited[j]:
    #         Candidate for next.
    #         Overlap is at least l.
    #         (Actually, if j starts with s, overlap is exactly l? 
    #          Wait, if curr ends with s and j starts with s, overlap is l.
    #          Could it be larger? No, because s is the suffix of curr.
    #          So overlap is l.)
    
    # This is much faster!
    # We only check permutations that share a prefix with a suffix of curr.
    
    # Let's implement this.
    
    best_order = None
    best_score = -1
    
    # We might want to try multiple starts or random restarts if time permits.
    # But let's start with one greedy pass.
    
    for start_idx in range(min(5, N)): # Try a few starts
        visited = [False] * N
        order = []
        curr = start_idx
        visited[curr] = True
        order.append(curr)
        
        while len(order) < N:
            curr_perm = all_perms[curr]
            best_next = -1
            best_ov = -1
            
            # Check suffixes of curr_perm
            # Longest first
            for l in range(n-1, 0, -1):
                s = tuple(curr_perm[-l:])
                if s in prefix_map:
                    candidates = prefix_map[s]
                    for j in candidates:
                        if not visited[j]:
                            # Overlap is l
                            if l > best_ov:
                                best_ov = l
                                best_next = j
                                # Can we break? 
                                # We want max overlap. If we found l, no other suffix of length < l can be better.
                                # But there might be another candidate with same l?
                                # Any is fine for greedy.
                                # But wait, is it possible that a shorter suffix leads to a better *future* path?
                                # Greedy doesn't care.
                                break
                    if best_ov == l:
                        break
            
            if best_next == -1:
                # No overlap found. Pick any unvisited.
                # This shouldn't happen if we cover all permutations.
                # Actually, if no suffix matches, overlap is 0.
                # We need to find ANY unvisited.
                # This is O(N).
                for j in range(N):
                    if not visited[j]:
                        best_next = j
                        best_ov = 0
                        break
            else:
                # Found best_next with overlap best_ov
                pass
            
            visited[best_next] = True
            order.append(best_next)
            curr = best_next
        
        # Calculate score (total overlap)
        score = 0
        for i in range(len(order) - 1):
            a = all_perms[order[i]]
            b = all_perms[order[i+1]]
            # Calculate overlap
            max_l = min(len(a), len(b))
            for l in range(max_l - 1, -1, -1):
                if a[-l:] == b[:l]:
                    score += l
                    break
        
        if score > best_score:
            best_score = score
            best_order = order[:]
            
        if time.time() > deadline - 10:
            break
    
    if best_order is None:
        # Fallback: standard order
        best_order = list(range(N))
        
    # Construct the word
    word = []
    for i, idx in enumerate(best_order):
        p = all_perms[idx]
        if i == 0:
            word.extend(p)
        else:
            prev_p = all_perms[best_order[i-1]]
            # Find overlap
            max_l = min(len(prev_p), len(p))
            ov = 0
            for l in range(max_l - 1, -1, -1):
                if prev_p[-l:] == p[:l]:
                    ov = l
                    break
            word.extend(p[ov:])
            
    return "".join(map(str, word))
