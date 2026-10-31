import time
import math
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Precompute powers of omega = exp(2*pi*i/3)
# We work in complex numbers.
# omega^0 = 1, omega^1 = -0.5 + i*sqrt(3)/2, omega^2 = -0.5 - i*sqrt(3)/2
# Instead of full complex FFT which might be overkill or slow for small n,
# we can compute the "Fourier energy" or a proxy for it.
# 
# For a vector v in {0,1,2}^n, the DFT coefficient at frequency k is:
# X[k] = sum_{v in set} omega^{k dot v}
#
# However, we are adding vectors one by one. We don't have the whole set.
# The prompt says: "vectors of {0,1,2}^n are added greedily in descending priority".
# The `priority` function is called for a single vector `v`.
# The evaluator maintains the set. We don't see the set.
# Therefore, we cannot compute the global DFT of the *set*.
# We can only compute properties of the *candidate vector*.
#
# Wait, the inspiration 1 mentions "global spectral representation by scoring vectors based on the magnitude of their Discrete Fourier Transform over Z_3^n".
# If we can't see the set, how do we use global properties?
# Perhaps the "vector" passed in is not just a single point, but the program is expected to be stateful?
# No, the interface is `priority(vector: tuple, n: int) -> float`.
# It is stateless.
#
# Let's re-read the inspiration: "scoring vectors based on the magnitude of their Discrete Fourier Transform over Z_3^n".
# A single vector in {0,1,2}^n is just a point. Its "Fourier Transform" isn't standard.
# Maybe it means: Treat the vector as a function on Z_3? No, it's in Z_3^n.
#
# Alternative interpretation:
# The "vector" is the candidate. We want to pick candidates that, when added to the *current set*, minimize the risk of forming a line.
# Since we don't see the set, we must assume the set is built by a process that we can approximate, or we use a property of the vector itself that correlates with "cap-ness".
#
# Actually, there is a known result: A set S is a cap set if it contains no 3 points x, x+d, x+2d.
# If we assume the greedy algorithm adds points, and we want to avoid lines, we should prefer points that are "far" from existing points? No, we don't know existing points.
#
# Let's look at Inspiration 1 more closely. It says "uses math, numpy, time, zlib; defines priority".
# Maybe the "vector" is actually a *frequency* vector? No, the task says "vectors of {0,1,2}^n".
#
# Let's reconsider the "Spectral" idea for a *single point*.
# In the context of the polynomial method (Ellenberg-Gijswijt), the size of a cap set is bounded by the slice rank of a certain tensor.
# The tensor involves terms like $\omega^{x_i y_i z_i}$.
#
# Perhaps the strategy is:
# Score a vector $v$ based on how "balanced" it is in the Fourier domain of the *single coordinate*?
# No, that's just entropy again.
#
# Let's try a different structural change: **Geometric Distance to Known Cap Sets**.
# We can precompute a few known small cap sets (for small n) or use a heuristic that approximates the "cap set density".
# For large n, the maximum cap set size is approx $2.756^n$.
# The density is $\rho = \log_3(2.756) \approx 0.91$.
# A random set has density $1/3^n$? No, a random subset of size $K$ has probability $3^{-2n}$ of containing a line?
#
# Let's look at the "Spectral" inspiration again.
# "scoring vectors based on the magnitude of their Discrete Fourier Transform over Z_3^n"
# If `vector` is a tuple of length n, it is a point in $\mathbb{Z}_3^n$.
# The DFT of a *function* $f: \mathbb{Z}_3^n \to \mathbb{C}$ is defined.
# But we have a point, not a function.
# Unless... the `vector` argument is actually a *mask* or a *frequency*?
# No, "vectors of {0,1,2}^n are added greedily".
#
# Maybe the inspiration implies that we should treat the *set* as the object, but since we can't, we simulate the effect.
#
# Let's try a **Greedy Conflict Avoidance via Hashing** strategy (Inspiration 2/3).
# This is a different family: instead of statistical entropy, use **combinatorial conflict detection**.
# We can maintain a global state (a set of hashes) of vectors that have been "rejected" or "considered"?
# No, the function is stateless.
#
# Wait, if the function is stateless, it cannot know what is already in the set.
# The evaluator calls `priority(v)` for each candidate `v` in a batch?
# Or does it call it for every possible vector?
# "vectors ... are added greedily in descending priority".
# This implies the evaluator has a list of all $3^n$ vectors, sorts them by `priority`, and adds them one by one if they don't form a line.
#
# If so, the `priority` function defines a total order on $\mathbb{Z}_3^n$.
# The quality of the resulting cap set depends on this order.
# We want an order such that the first $K$ vectors (where $K$ is the max cap set size) form a cap set.
# This is equivalent to finding a cap set that is "lexicographically first" (or first by our priority) in $\mathbb{Z}_3^n$.
#
# So, the problem reduces to: **Define a priority function $P: \mathbb{Z}_3^n \to \mathbb{R}$ such that the greedy algorithm produces a large cap set.**
#
# A known good cap set construction is the **Ellenberg-Gijswijt** set, which is defined by a polynomial condition.
# Specifically, $S = \{ x \in \mathbb{Z}_3^n : f(x) = 0 \}$ for some polynomial $f$ of degree $d < 2n/3$.
# If we can identify vectors that are "likely" to be in a large cap set, we should prioritize them.
#
# One such construction is based on the **Slice Rank** method.
# Another is the **Ellenberg-Gijswijt** bound, which uses the polynomial $\prod (1 - x_i^2)$.
#
# Let's try a strategy based on **Polynomial Evaluation**.
# The set $S = \{ x \in \mathbb{Z}_3^n : \sum_{i=1}^n x_i^2 = 0 \pmod 3 \}$?
# In $\mathbb{Z}_3$, $0^2=0, 1^2=1, 2^2=1$.
# So $x_i^2$ is 0 if $x_i=0$, and 1 if $x_i \in \{1,2\}$.
# $\sum x_i^2 \equiv 0 \pmod 3$ means the number of non-zero elements is a multiple of 3.
# This defines a specific set. Is it a cap set?
# If $x, x+d, x+2d$ are in the set, do they form a line?
# This is a specific algebraic structure.
#
# Let's test this hypothesis: **Prioritize vectors based on the sum of squares modulo 3.**
# Or more generally, use a quadratic form.
# The best known cap sets for small n are often related to quadratic forms.
#
# Strategy:
# 1. Calculate $Q(v) = \sum_{i=1}^n v_i^2 \pmod 3$.
# 2. Prioritize vectors where $Q(v) = 0$.
# 3. Tie-break with entropy or hash.
#
# Why might this work?
# The set $\{ v : Q(v) = 0 \}$ is an algebraic variety.
# If this variety contains no lines, it's a cap set.
# Even if it contains lines, prioritizing it might lead to a better greedy result than random/entropy because it exploits the algebraic structure of $\mathbb{Z}_3^n$.
#
# Let's refine:
# Use a weighted quadratic form.
# $Q(v) = \sum_{i=1}^n w_i v_i^2 \pmod 3$.
# If we set all $w_i = 1$, we get the "weight divisible by 3" set.
#
# Let's check if "weight divisible by 3" is a cap set.
# Take $n=2$. Vectors in $\{0,1,2\}^2$ with weight (sum of squares) $\equiv 0 \pmod 3$.
# Weights:
# 00: 0 (Yes)
# 10: 1 (No)
# 20: 1 (No)
# 01: 1 (No)
# 02: 1 (No)
# 11: 2 (No)
# 12: 2 (No)
# 21: 2 (No)
# 22: 2 (No)
# Only 00 is in the set. Size 1. Max cap set size for n=2 is 4 (e.g., {00, 11, 22, 01}? No, 00,11,22 is a line. {00, 12, 21} is a line. Max is 4: {00, 11, 20, 12}? 00,11,22 is line.
# Actually, max cap set size for n=2 is 4. Example: {00, 11, 20, 02}?
# 00, 11, 22 is a line.
# 00, 10, 20 is a line.
# 00, 01, 02 is a line.
# 11, 12, 10 is a line.
# Let's try {00, 12, 21, 01}?
# 00, 01, 02 (no 02).
# 12, 21, 10 (no 10).
# 21, 01, 11 (no 11).
# 01, 12, 20 (no 20).
# Is {00, 12, 21, 01} a cap set?
# Lines:
# 00, 12, 21? 00+12=12, 12+12=24=21. Yes, 00,12,21 is a line.
# So {00, 12, 21} is not a cap set.
#
# The "sum of squares" heuristic might be too restrictive or not optimal.
#
# Let's look at the **Ellenberg-Gijswijt** construction again.
# They use the polynomial $f(x) = \prod_{i=1}^n (1 - x_i^2)$.
# The set $S = \{ x : f(x) \neq 0 \}$? No, $f(x)$ is 0 if any $x_i \neq 0$.
# So $f(x) \neq 0$ implies $x=0$. That's size 1.
# They use a different polynomial for the bound.
#
# However, there is a known family of cap sets: **The AGW (Alon-Gijswijt-Wigderson) construction**?
# Or simply: **Prioritize vectors with low "line density" in their neighborhood**.
#
# Let's go back to **Spectral/FFT**.
# If we assume the greedy algorithm picks the top K vectors, and we want them to be a cap set, we want the top K vectors to be "spread out" in a way that avoids lines.
# A line in $\mathbb{Z}_3^n$ is an arithmetic progression.
# In the Fourier domain, lines correspond to specific frequencies.
#
# Let's try a **Randomized Projection** strategy.
# Map $\mathbb{Z}_3^n$ to $\mathbb{R}^d$ using a random linear map, and prioritize vectors based on their norm or angle?
# No, that's just another hash.
#
# Let's try **Minimizing the Sum of Squared Distances to the Origin**?
# No.
#
# Let's look at the **Inspiration 1** again: "magnitude of their Discrete Fourier Transform over Z_3^n".
# If `vector` is a point $x$, its DFT is not defined.
# BUT, if we view the *set of points selected so far* as a function $f: \mathbb{Z}_3^n \to \{0,1\}$, then we can compute its DFT.
# Since we can't access the set, maybe the "vector" passed to `priority` is NOT a single point, but a **frequency vector**?
# No, the task says "vectors of {0,1,2}^n are added greedily".
#
# Is it possible that the `vector` is a **mask** indicating which coordinates are fixed? No.
#
# Let's assume the standard interpretation: `vector` is a candidate point $x \in \mathbb{Z}_3^n$.
#
# New Idea: **Use the "Cap Set Density" of the Hamming Ball around x**.
# We can precompute the maximum cap set size for small radii?
# No, $n$ can be large.
#
# Let's try a **Different Optimization Method**: **Simulated Annealing-like Priority**.
# We can't use temperature if stateless.
#
# How about **Lexicographic Order based on a Permutation**?
# If we permute the coordinates, the structure of lines changes.
# If we prioritize vectors based on their value in a permuted coordinate system, we might break symmetries that lead to early line formation.
#
# Strategy:
# 1. Define a fixed permutation of coordinates $\pi$.
# 2. Sort coordinates by $\pi$.
# 3. Prioritize vectors that are "small" in this permuted lexicographic order?
#
# This is still a heuristic.
#
# Let's go with the **Algebraic Structure** approach.
# The set of points $x$ such that $\sum_{i=1}^n x_i = 0 \pmod 3$ is a hyperplane.
# A hyperplane in $\mathbb{Z}_3^n$ contains lines.
# The set of points $x$ such that $\sum_{i=1}^n x_i^2 = 0 \pmod 3$ is a quadric.
# Quadrics in $\mathbb{Z}_3^n$ can be cap sets if the form is non-degenerate?
#
# Let's test the **Quadratic Form** strategy.
# $Q(x) = \sum_{i=1}^n x_i^2 \pmod 3$.
# Prioritize $x$ where $Q(x) = 0$.
# Tie-break: $Q(x) = 1$, then $Q(x) = 2$.
#
# Why?
# If the greedy algorithm picks all $x$ with $Q(x)=0$ first, does it form a cap set?
# For $n=1$, $Q(0)=0, Q(1)=1, Q(2)=1$. Set $\{0\}$. Size 1. Max 2? No, max cap set size for n=1 is 2? {0,1} is not a line. {0,2} is not a line. {1,2} is not a line. Max size 2.
# My set $\{0\}$ has size 1. Bad.
#
# What if we prioritize $Q(x) \neq 0$?
# Set $\{1, 2\}$. Size 2. Good.
#
# For $n=2$, $Q(x) = x_1^2 + x_2^2 \pmod 3$.
# $Q(00)=0$.
# $Q(10)=1, Q(20)=1, Q(01)=1, Q(02)=1$.
# $Q(11)=2, Q(12)=2, Q(21)=2, Q(22)=2$.
# If we prioritize $Q(x) \neq 0$, we pick $\{10, 20, 01, 02, 11, 12, 21, 22\}$.
# Does this set contain a line?
# Line: 10, 11, 12? $10+11=21 \neq 12$. No.
# Line: 10, 20, 00? 00 not in set.
# Line: 11, 12, 10? $11+12=23=20 \neq 10$. No.
# Line: 11, 22, 00? 00 not in set.
# Line: 12, 21, 10? $12+21=33=00 \neq 10$. No.
# It seems $\{x : Q(x) \neq 0\}$ might be a cap set for $n=2$. Size 8. Max size 4?
# Wait, max cap set size for $n=2$ is 4.
# So $\{x : Q(x) \neq 0\}$ is NOT a cap set. It must contain a line.
# Let's find it.
# Points: 10, 20, 01, 02, 11, 12, 21, 22.
# Line: 10, 21, 12?
# $10 + 21 = 31 = 01 \neq 12$.
# $10 + 12 = 22$. So 10, 12, 22 is a line?
# $10, 12, 22$: $12-10=02$. $22-12=10$. Not an AP.
# AP: $x, x+d, x+2d$.
# $10, 12, 22$: $12-10=02$. $22-12=10$. $02 \neq 10$.
#
# Let's check 11, 22, 00 (00 not in set).
# 11, 20, 02? $20-11=09=00$. $02-20=12$. No.
#
# Actually, the max cap set size for $n=2$ is 4.
# The set $\{x : Q(x) \neq 0\}$ has size 8. It MUST contain a line.
#
# So prioritizing $Q(x) \neq 0$ is bad because it picks too many points that form lines.
#
# Let's try prioritizing **$Q(x) = 0$** again.
# For $n=2$, size 1. Too small.
#
# What if we use a **Linear Form**?
# $L(x) = \sum_{i=1}^n c_i x_i \pmod 3$.
# The set $\{x : L(x) = 0\}$ is a hyperplane, which contains lines.
#
# It seems algebraic sets defined by low-degree polynomials are not cap sets (they contain lines).
# Cap sets are *avoiding* lines.
#
# So we should prioritize vectors that are **NOT** in the algebraic sets that contain lines?
# i.e., Prioritize $x$ where $Q(x) \neq 0$?
# But we saw that for $n=2$, $Q(x) \neq 0$ has size 8, which is larger than the max cap set size 4.
# So the greedy algorithm will pick the first 4 points from this set.
# Will the first 4 points form a cap set?
# If the priority is deterministic, the first 4 points are fixed.
# If we can choose the priority such that the first 4 points are a cap set, we win.
#
# So the strategy is: **Find a priority function such that the top $K$ vectors form a cap set.**
# This is equivalent to **Embedding a known cap set at the top of the priority order**.
#
# How to find a known cap set?
# For small $n$, we can precompute the max cap set.
# For large $n$, we can use the **Ellenberg-Gijswijt** construction, which gives a set of size $\approx 2.756^n$.
#
# The EG set is defined by $S = \{ x \in \mathbb{Z}_3^n : f(x) = 0 \}$ where $f$ is a specific polynomial.
# If we can evaluate $f(x)$ quickly, we can prioritize $x$ where $f(x)=0$.
#
# The polynomial is $f(x) = \prod_{i=1}^n (1 - x_i^2)$.
# $f(x) = 0$ if any $x_i \neq 0$.
# So $S = \{0\}$. Size 1. This is not the EG set.
#
# The EG bound uses a different argument. The set is not explicitly given by a simple zero set of a low-degree polynomial in the same way.
#
# However, there is a construction by **Bukh and Cox**.
#
# Let's try a **Hybrid Strategy**:
# 1. Use the **Quadratic Form** $Q(x) = \sum x_i^2 \pmod 3$.
# 2. Prioritize $x$ based on $Q(x)$.
# 3. Specifically, prioritize $x$ where $Q(x) = 1$ or $2$?
#
# Let's look at the **Inspiration 1** again. "Spectral representation".
# Maybe it means: Compute the **Power Spectrum** of the vector?
# $P(k) = | \sum_{i=1}^n x_i \omega^{ki} |^2$?
# This is the DFT of the vector $x$ treated as a sequence.
#
# Let's implement **DFT of the Vector**.
# $X[k] = \sum_{i=0}^{n-1} x_i \omega^{ki}$.
# Score = $\sum_{k=0}^{n-1} |X[k]|^2$.
# By Parseval's theorem, $\sum |X[k]|^2 = n \sum x_i^2$.
# So this is just proportional to the sum of squares.
#
# What if we use **Higher Order Spectra**?
#
# Let's try **Minimizing the Autocorrelation**.
# A cap set should have low autocorrelation to avoid lines.
# Autocorrelation $R(\tau) = \sum_{i} x_i x_{i+\tau}$.
#
# Let's go with a **Simple Algebraic Heuristic**:
# Prioritize vectors where the **Sum of Coordinates** is balanced.
# $S(x) = \sum x_i \pmod 3$.
# Prioritize $S(x) = 0$.
#
# This is a hyperplane. It contains lines.
#
# Okay, I will implement the **DFT Magnitude** strategy as suggested by Inspiration 1, assuming it refers to the DFT of the vector components.
# Score = $\sum_{k=0}^{n-1} |X[k]|^2$ is just sum of squares.
# Let's use the **Variance of the DFT Magnitudes**.
# A "random" vector has flat DFT.
# A "structured" vector has peaks.
# Cap sets are "unstructured" in the sense of avoiding lines, but they are structured in the sense of being a set.
#
# Let's try: **Score = -Variance(DFT_Magnitudes)**.
# Prefer flat DFTs (random-like).
#
# This is similar to entropy but in the frequency domain.
#
# Implementation:
# 1. Compute DFT of the vector (using numpy if n is small, or manual for speed).
# 2. Compute magnitudes.
# 3. Compute variance.
# 4. Return negative variance (or 1/variance).
#
# This is a structural change from time-domain entropy to frequency-domain flatness.

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    # Convert to numpy array for FFT
    # n can be up to 100-200?
    # FFT is O(n log n).
    # If n is large, this might be slow.
    # But 300s budget allows many calls.
    
    # If n is very small, FFT is overkill.
    # Let's use a simple heuristic for small n.
    
    if n < 10:
        # Fallback to entropy
        c0 = 0
        c1 = 0
        c2 = 0
        for v in vector:
            if v == 0: c0 += 1
            elif v == 1: c1 += 1
            else: c2 += 1
        h = 0.0
        for c in (c0, c1, c2):
            if c > 0:
                p = c / n
                h -= p * math.log2(p)
        return h
    
    # For larger n, use Spectral Flatness
    try:
        x = np.array(vector, dtype=np.float64)
        # Compute FFT
        # Note: vector elements are 0, 1, 2.
        # FFT expects complex or real.
        X = np.fft.fft(x)
        magnitudes = np.abs(X)
        
        # Spectral Flatness Measure (SF)
        # SF = Geometric Mean / Arithmetic Mean
        # High SF means flat spectrum (random-like).
        # Low SF means peaky spectrum (structured).
        
        # To avoid log(0), add epsilon
        eps = 1e-9
        geo_mean = np.exp(np.mean(np.log(magnitudes + eps)))
        arith_mean = np.mean(magnitudes)
        
        sf = geo_mean / arith_mean
        
        # Prefer high SF
        return float(sf)
        
    except Exception:
        return 0.0
