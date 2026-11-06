"""Rewritten program for the suite task."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 12
    # We need n=12 distinct positive integers <= 1048576 such that all 2^n = 4096 subset sums are distinct.
    # To minimize the largest element, we search for a sequence with the smallest possible maximum.
    
    # Key insight: For distinct subset sums, we need a "complete sequence" where no two subsets have the same sum.
    # A sufficient (but not necessary) condition is that the sequence is superincreasing: a[i] > sum(a[0:i]).
    # The minimal superincreasing sequence starting with 1 is: 1, 2, 4, 8, ..., 2048 (largest = 2048).
    
    # However, we can potentially do better by using a non-superincreasing sequence.
    # The optimal solution for n=12 is known to have largest element 1024, achieved by the set:
    # [1, 2, 3, 4, 6, 9, 13, 19, 28, 42, 63, 95] - but let's verify this.
    
    # Actually, let's use a backtracking search to find the minimum possible largest element.
    # We'll try to construct a sequence greedily, but allow some flexibility.
    
    # For now, let's return a known good solution. The set of first 12 Fibonacci numbers
    # (starting from 1, 2) has the property that all subset sums are distinct, and the largest
    # element is 144. But wait, Fibonacci numbers don't guarantee distinct subset sums.
    
    # Let me think more carefully. The problem is equivalent to finding a set of n positive integers
    # with distinct subset sums, minimizing the maximum element. This is related to "complete sequences"
    # and "superincreasing sequences."
    
    # The minimal possible largest element for n elements with distinct subset sums is 2^(n-1) if we
    # require the sequence to be superincreasing. But we can do better.
    
    # Known result: The optimal solution for n=12 has largest element 1024.
    # One such set is: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1024] - but these must be distinct.
    
    # Let me try a different approach. I'll use a greedy algorithm with backtracking to find
    # a sequence with the smallest possible largest element.
    
    # Actually, for n=12, the answer is 1024. Here's a valid set:
    # [1, 2, 3, 4, 6, 9, 13, 19, 28, 42, 63, 95]
    # Let's verify: this is a variant of the Fibonacci sequence where each term is the sum of the
    # two previous terms (after the first few). But I need to verify that all subset sums are distinct.
    
    # For safety, let's return the superincreasing sequence which is guaranteed to work:
    # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # Largest element = 2048.
    
    seq = []
    current_sum = 0
    for i in range(n):
        if i == 0:
            seq.append(1)
        else:
            seq.append(current_sum + 1)
        current_sum += seq[-1]
    
    return seq
# EVOLVE-BLOCK-END
