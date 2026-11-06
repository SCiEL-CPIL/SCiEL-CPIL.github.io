import time
import random
from math import gcd

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    N = 256
    is_prime = [True]*(N+1)
    is_prime[0]=is_prime[1]=False
    for i in range(2,int(N**0.5)+1):
        if is_prime[i]:
            for j in range(i*i,N+1,i):
                is_prime[j]=False
    primes = [i for i in range(2,N+1) if is_prime[i]]

    def evaluate(A,B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a+b)
        lst = list(sums)
        for i in range(len(lst)):
            for j in range(i+1,len(lst)):
                if gcd(lst[i],lst[j])!=1:
                    return 0
        return len(lst)

    # Seed
    best = ([1,7,13],[4,10,16])
    best_card = evaluate(best[0],best[1])
    if best_card<=0:
        best_card = 5  # seed is known valid

    rng = random.Random(20240517)

    # Exhaustive sum-grid search for cardinality 9 (all-prime)
    for P in primes:
        if time.time() > deadline:
            break
        S = [x for x in range(-127,128) if x!=0 and 2<=P+x<=N and is_prime[P+x]]
        if len(S)<4:
            continue
        S2 = S  # offsets
        found=False
        for r2 in S2:
            if found: break
            if time.time() > deadline: break
            for r3 in S2:
                if found: break
                if r2==r3: continue
                # valid c: in S, distinct from r2,r3 (not required but for 9 distinctness we check later), cross prime
                valid_c=[]
                for c in S2:
                    if is_prime[P+r2+c] and is_prime[P+r3+c]:
                        valid_c.append(c)
                if len(valid_c)<2:
                    continue
                # pick c2,c3 from valid_c, distinct
                for ci in range(len(valid_c)):
                    for cj in range(ci+1,len(valid_c)):
                        c2=valid_c[ci]; c3=valid_c[cj]
                        nine=[P,P+c2,P+c3,P+r2,P+r3,P+r2+c2,P+r2+c3,P+r3+c2,P+r3+c3]
                        if len(set(nine))!=9:
                            continue
                        r_list=[0,r2,r3]; c_list=[0,c2,c3]
                        rmin=min(r_list); rmax=max(r_list); cmin=min(c_list); cmax=max(c_list)
                        lo=max(1,1-rmin,P-128,P-128+cmax)
                        hi=min(128,128-rmax,P-1,P-1+cmin)
                        if lo<=hi:
                            a1=rng.randint(lo,hi)
                            A=[a1,a1+r2,a1+r3]
                            B=[P-a1,P-a1+c2,P-a1+c3]
                            if all(1<=x<=128 for x in A) and all(1<=x<=128 for x in B) and len(set(A))==3 and len(set(B))==3:
                                card=evaluate(A,B)
                                if card==9:
                                    best=(A[:],B[:]); best_card=9; found=True
                                    break
                    if found: break
                if found: break
            if found: break
        if found:
            break

    if best_card>=9 or time.time()>deadline:
        return best

    # Fallback: heavy random sampling + local search to maximize cardinality
    def rand_set():
        while True:
            s=[rng.randint(1,128) for _ in range(3)]
            if len(set(s))==3:
                return s

    it=0
    while time.time()<deadline:
        it+=1
        A=rand_set(); B=rand_set()
        card=evaluate(A,B)
        if card>best_card:
            best=(A[:],B[:]); best_card=card
            if best_card>=9:
                break
        # local search: try single mutations
        if card>0:
            # try improving
            for _ in range(20):
                if time.time()>deadline: break
                A2=A[:]; B2=B[:]
                if rng.random()<0.5:
                    A2[rng.randint(0,2)]=rng.randint(1,128)
                else:
                    B2[rng.randint(0,2)]=rng.randint(1,128)
                if len(set(A2))<3 or len(set(B2))<3: continue
                c2=evaluate(A2,B2)
                if c2>best_card:
                    best=(A2[:],B2[:]); best_card=c2
                    A,B=A2,B2
                    if best_card>=9: break

    return best
