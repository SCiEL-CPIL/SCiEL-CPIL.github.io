"""
Maximum number of pairwise-disjoint infinite residue classes with distinct moduli in 2..128.

Two residue classes (r1, m1) and (r2, m2) are disjoint iff
    r1 % gcd(m1, m2) != r2 % gcd(m1, m2)
because they intersect iff r1 ≡ r2 (mod gcd(m1, m2)).

Strategy:
- Build a maximum set S of moduli from {2..128} for which we can assign residues
  satisfying all pairwise gcd-constraints.
- Use backtracking with constraint propagation (forward checking) to search over
  moduli and residues. To keep the search tractable, we:
    1. Precompute all prime powers and group moduli by their "radical" (product of
       distinct prime factors). Moduli that share a prime factor will have gcd > 1
       and thus impose a real constraint.
    2. Use a greedy + local-search hybrid to find a large feasible set, then verify.
- Actually, a cleaner approach: model as a graph where vertices are moduli, edge
  exists if gcd(m_i, m_j) > 1 (i.e., they interact). For coprime pairs, no constraint
  is needed (any residues work since gcd=1 means r1%1 = r2%1 = 0 always... wait, 
  gcd=1 means the condition is r1 % 1 != r2 % 1, but 0 != 0 is false! So coprime
  moduli ALWAYS intersect? No wait:

Let me re-derive: Two arithmetic progressions r1 + k*m1 and r2 + k*m2 intersect iff
there exist a,b >= 0 such that r1 + a*m1 = r2 + b*m2, i.e., r1 - r2 = b*m2 - a*m1.
This has a solution iff gcd(m1, m2) | (r1 - r2). So they are DISJOINT iff
gcd(m1, m2) does NOT divide (r1 - r2), i.e., r1 % g != r2 % g where g = gcd(m1,m2).

When g = 1: r1 % 1 = 0 and r2 % 1 = 0, so 0 != 0 is FALSE. They are NOT disjoint.
So coprime moduli always have intersecting residue classes! This is a key insight.

Therefore, to have disjoint residue classes, we NEED gcd(m1, m2) > 1 for every pair,
AND we need to assign residues such that r1 % gcd(m1,m2) != r2 % gcd(m1,m2).

This means we want a set of moduli where every pair shares a common prime factor,
and we can assign residues to satisfy all the modular inequalities.

This is equivalent to: find the largest subset S ⊆ {2..128} and assignment of residues
such that for all i≠j in S: r_i ≢ r_j (mod gcd(m_i, m_j)).

Key structural observation: If we pick moduli that are all powers of 2 (2,4,8,16,32,64,128),
then gcd(2^a, 2^b) = 2^min(a,b). We need r_i % 2^min(a,b) != r_j % 2^min(a,b) for all pairs.
If we set r_i = 2^{a-1} (i.e., r_i = m_i/2), then for m_i=2^a, m_j=2^b with a<b:
gcd = 2^a, r_i % 2^a = 2^{a-1}, r_j % 2^a = 2^{b-1} % 2^a = 0 (since b-1 >= a when b>a... 
wait, if b > a, then 2^{b-1} % 2^a = 0 if b-1 >= a, i.e., b >= a+1. Yes.
So r_i % 2^a = 2^{a-1} != 0 = r_j % 2^a. Works!

So {2,4,8,16,32,64,128} with residues {1,2,4,8,16,32,64} gives 7 disjoint classes.

Can we do better by mixing primes? If we include a multiple of 3, say 3, then gcd(2,3)=1,
so they MUST intersect. So we can't mix different prime bases at all.

What about all multiples of 2? E.g., {2,4,6,8,10,...}. gcd(2,6)=2, gcd(4,6)=2, gcd(6,10)=2, etc.
But we need r_i % 2 != r_j % 2 for all pairs with gcd=2. That means all residues must have
the same parity... but then r_i % 2 = r_j % 2 for all pairs, so they'd all intersect.

Actually if gcd(m_i, m_j) = 2 for many pairs, we need r_i % 2 != r_j % 2, which is impossible
for more than 2 elements (pigeonhole: only 0 or 1 mod 2).

So the power-of-2 chain is special because the gcds are nested: gcd(2^a, 2^b) = 2^{min(a,b)},
and the constraints are hierarchical.

Let me think about whether we can get more than 7. The moduli must all share a common prime
factor (otherwise two would be coprime and intersect). WLOG say all are even. Then we need
residues such that for each pair, r_i ≢ r_j (mod gcd(m_i, m_j)).

For powers of 2: we got 7. Can we add a non-power-of-2 even number to {2,4,8,16,32,64,128}?
Say add 12. gcd(12, 2)=2: need r_12 % 2 != 1 % 2 = 1, so r_12 even.
gcd(12, 4)=4: need r_12 % 4 != 2 % 4 = 2.
gcd(12, 8)=4: need r_12 % 4 != 4 % 4 = 0.
So r_12 % 4 != 2 and r_12 % 4 != 0, so r_12 % 4 must be 1 or 3. But r_12 even means r_12 % 4 is 0 or 2. Contradiction.

Try adding 20: gcd(20,2)=2: r_20 even. gcd(20,4)=4: r_20%4 != 2. gcd(20,8)=4: r_20%4 != 0.
Same issue: r_20%4 must be 1 or 3, but even means 0 or 2. Contradiction.

In general, any even number not a power of 2 will have gcd with some 2^k equal to 2^a where a >= 1,
and the constraints from multiple powers of 2 will force contradictions.

So the maximum is 7 (powers of 2 from 2 to 128).

Let me verify: 2,4,8,16,32,64,128. That's 7 moduli. Residues: 1,2,4,8,16,32,64.
Check a few: (2,4): gcd=2, 1%2=1, 2%2=0, 1!=0 OK.
(2,128): gcd=2, 1%2=1, 64%2=0, OK.
(4,128): gcd=4, 2%4=2, 64%4=0, OK.
(64,128): gcd=64, 32%64=32, 64%64=0, OK.

All good. Score should be 7.
"""


def construct():
    # Powers of 2 from 2^1 to 2^7 = 128
    powers_of_2 = [2**i for i in range(1, 8)]  # [2, 4, 8, 16, 32, 64, 128]
    # Residues: m/2 for each m
    residues = [m // 2 for m in powers_of_2]
    
    result = [[r, m] for r, m in zip(residues, powers_of_2)]
    return result
