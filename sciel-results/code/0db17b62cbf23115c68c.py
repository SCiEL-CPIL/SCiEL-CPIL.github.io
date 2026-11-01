def construct():
    start_time = time.time()
    # ... existing code ...
    elapsed = time.time() - start_time
    if elapsed > 135:
        return {"U": U, "V": V, "W": W}
    # ... rest ...
