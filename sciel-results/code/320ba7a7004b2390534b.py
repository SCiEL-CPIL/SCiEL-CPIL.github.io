"""Find the smallest n in [1, 1000000] with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START

def construct():
    """Return the smallest n in [1,1000000] such that 120 divides phi(n).
    
    The answer is 240:
    240 = 2^4 * 3 * 5
    phi(240) = 240 * (1 - 1/2) * (1 - 1/3) * (1 - 1/5)
            = 240 * 1/2 * 2/3 * 4/5
            = 240 * 8/30
            = 240 * 4/15
            = 64
    Wait, that's not divisible by 120. Let me recalculate.
    
    phi(240) = 240 * (1/2) * (2/3) * (4/5) = 240 * 8/30 = 64. Not divisible by 120.
    
    Let me check n=241: 241 is prime, phi(241) = 240. 240/120 = 2. Yes!
    
    But is there a smaller n? We need phi(n) divisible by 120.
    phi(n) must be at least 120.
    
    Let me think about what values of n give phi(n) >= 120.
    For n=1: phi=1
    For primes p: phi(p)=p-1. So p-1 divisible by 120 means p = 121, 241, 361, 481, 601, 721, 841, 961, 1081, 1201, ...
    121 = 11^2, not prime. 241 is prime! phi(241)=240, 240/120=2. ✓
    
    But maybe a composite n < 241 works?
    phi(n) divisible by 120 means 120 | phi(n).
    
    Let me check some candidates:
    n=240: phi(240)=64. No.
    n=241: phi(241)=240. Yes! 240%120=0. ✓
    
    Is there anything smaller? Let's check values where phi(n) could be divisible by 120.
    phi(n) must be a multiple of 120: 120, 240, 360, ...
    
    For phi(n)=120: n could be 121 (not prime, phi=110), 187 (11*17, phi=160), 259 (7*37, phi=240), 341 (11*31, phi=300), 377 (13*29, phi=336), 481 (13*37, phi=420), 561 (3*11*17, phi=320), 649 (11*59, phi=580), 671 (11*61, phi=600), 703 (19*37, phi=636), 961 (31^2, phi=930), 1105 (5*13*17, phi=768)...
    
    Actually let me just be systematic. The smallest n with phi(n) divisible by 120:
    
    n=241: phi(241) = 240, 240 % 120 = 0. ✓
    
    Can we do better? Let's check if any n < 241 has phi(n) divisible by 120.
    
    phi(n) divisible by 120 = 2^3 * 3 * 5.
    
    For phi(n) to be divisible by 120, n must have enough prime factors to make phi(n) large enough and with the right divisibility.
    
    Let me just return 241 since I've verified it works, and checking all n < 241 would be tedious but let me reason:
    
    For n < 241, phi(n) < n < 241. So phi(n) can only be 120 (since next multiple is 240, and if phi(n)=240 then n>=241 since phi(n)<n for n>2... wait no, phi(n) can equal n-1 only for primes, and for composite n, phi(n) <= n/2 roughly... actually phi(n) < n for all n > 1).
    
    So for n < 241, phi(n) < 241. The only multiple of 120 less than 241 is 120 itself.
    So we need phi(n) = 120 for some n < 241.
    
    Values of n with phi(n) = 120:
    - If n is prime: n-1=120, n=121=11^2, not prime.
    - n = p*q where p,q are distinct primes: (p-1)(q-1)=120.
      Pairs (a,b) with a*b=120, a=p-1, b=q-1, p=a+1 prime, q=b+1 prime, p<q:
      (1,120): p=2, q=121=11^2 no
      (2,60): p=3, q=61 prime. n=3*61=183. phi(183)=(3-1)(61-1)=2*60=120. ✓
      (3,40): p=4 no
      (4,30): p=5, q=31 prime. n=5*31=155. phi(155)=4*30=120. ✓
      (5,24): p=6 no
      (6,20): p=7, q=21=3*7 no
      (8,15): p=9 no
      (10,12): p=11, q=13 prime. n=11*13=143. phi(143)=10*12=120. ✓
      (12,10): same as above
      So n=143, 155, 183 all have phi=120.
      
    The smallest is n=143. Let me verify: 143=11*13. phi(143)=(11-1)*(13-1)=10*12=120. 120%120=0. ✓
    
    So the answer is 143!
    """
    return 143

# EVOLVE-BLOCK-END
