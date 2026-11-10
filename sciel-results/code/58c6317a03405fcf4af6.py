<<<<<<< SEARCH
if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
=======
if __name__ == "__main__":
    start = time.time()
    e = construct()
    elapsed = time.time() - start
    print(f"Edges: {len(e)} (time: {elapsed:.1f}s)")
    if len(e) > 0:
        valid = _verify_no_c6(e)
        print(f"Valid (no C6): {valid}")
>>>>>>> REPLACE
